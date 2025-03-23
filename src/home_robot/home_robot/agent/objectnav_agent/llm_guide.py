from openai import OpenAI
from matplotlib import colors
import base64
import torch
import copy
import skimage
import cv2
import numpy as np
from PIL import Image
import io
from home_robot.mapping.semantic.constants import MapConstants as MC
import re

def map_postprocess(semantic_map, margin=15):
    # crop the semantic map, avoiding overmuch unknown area C H W
    # return the map with its size
    sum_in_channel = torch.sum(semantic_map, dim=0)
    map_loc = torch.nonzero((abs(sum_in_channel-sum_in_channel[0,0])>1e-4).int())

    values, indices = torch.min(map_loc, dim=0)
    min_y, min_x = map_loc[indices[0],0], map_loc[indices[1],1]
    values, indices = torch.max(map_loc, dim=0)
    max_y, max_x = map_loc[indices[0],0], map_loc[indices[1],1]

    min_x = (min_x-margin) if min_x >= margin else torch.tensor(0)
    min_y = (min_y-margin) if min_y >= margin else torch.tensor(0)
    max_x = (max_x+margin) if max_x <= (sum_in_channel.shape[0]-margin) else torch.tensor(sum_in_channel.shape[0])
    max_y = (max_y+margin) if max_y <= (sum_in_channel.shape[1]-margin) else torch.tensor(sum_in_channel.shape[1])
    
    semantic_map_cropped = copy.deepcopy(semantic_map[:, min_y:max_y, min_x:max_x])

    return semantic_map_cropped, [min_y, min_x, max_y, max_x]


def buildmap(final_global_map_input):
    final_global_map = copy.deepcopy(final_global_map_input)
    final_global_map = final_global_map.cpu()
    single_channel_map = torch.zeros_like(final_global_map[0,0,:,:])
    semantic_map = torch.stack((single_channel_map, single_channel_map, single_channel_map)) # C H W
    semantic_map = semantic_map.permute(1,2,0).double() # H W C
    
    unknown_rgb = colors.to_rgb('white') # entire map
    unknown_rgb_tensor = torch.tensor(unknown_rgb).double()
    semantic_map[:,:,:] = unknown_rgb_tensor

    free_rgb = colors.to_rgb('lightgreen') # explored map
    free_map = copy.deepcopy(final_global_map[0,MC.EXPLORED_MAP,:,:]) > 0.5
    semantic_map[free_map,:] = torch.tensor(free_rgb).double()    

    history_rgb = colors.to_rgb('lightskyblue') # traj
    history_rgb_tensor = torch.tensor(history_rgb).double()
    historical_area = copy.deepcopy(final_global_map[0,MC.VISITED_MAP,:,:]) > 0.5
    historical_area = skimage.morphology.binary_dilation(historical_area, skimage.morphology.disk(1))
    # dilate ?
    # historical_area = skimage.morphology.binary_dilation(historical_area,skimage.morphology.disk(1))
    semantic_map[historical_area, : ] = history_rgb_tensor

    obstacle_rgb = colors.to_rgb('dimgrey')
    obstacle_rgb_tensor = torch.tensor(obstacle_rgb).double()
    obstacle_area = copy.deepcopy(final_global_map[0,MC.OBSTACLE_MAP,:,:]) > 0.5
    obstacle_area = skimage.morphology.binary_dilation(obstacle_area, skimage.morphology.disk(1))
    semantic_map[obstacle_area,:] = obstacle_rgb_tensor

    agent_location_rgb = colors.to_rgb('darkgoldenrod')
    agent_location_rgb_tensor = torch.tensor(agent_location_rgb).double()
    agent_location_area = copy.deepcopy(final_global_map[0,MC.CURRENT_LOCATION,:,:]) > 0.5
    agent_location_area = skimage.morphology.binary_dilation(agent_location_area, skimage.morphology.disk(1))
    semantic_map[agent_location_area,:] = agent_location_rgb_tensor

    # resize ?
    # resize = 4
    # semantic_map_sr = cv2.resize(torch.flip(copy.deepcopy(semantic_map)*255,dims=[2]).numpy(), (semantic_map.shape[0]*resize, semantic_map.shape[1]*resize), interpolation=cv2.INTER_CUBIC)
    # semantic_map_sr_tensor = torch.flip(torch.from_numpy(semantic_map_sr.copy()), dims=[2]) # H W C , RGB
    # semantic_map_sr_tensor = semantic_map_sr_tensor/255
    
    # crop ?
    map_cropped, _ = map_postprocess(semantic_map.permute(2, 0, 1))
    save_map(map_cropped.permute(1, 2, 0), "/home/home-robot/datadump/images/local_img/semantic_map.png")
    return semantic_map # H W C


def save_map(semantic_map, path):
    try:
        semantic_map = semantic_map.cpu().numpy()
    except:
        print("")
    semantic_map = (semantic_map * 255).astype('uint8')

    image = Image.fromarray(semantic_map)
    image.save(path)

def LLMspatialreasoning(semantic_map, seq_lmb, goal):

    client = OpenAI(base_url="https://api3.apifans.com/v1", api_key="sk-moWc9vygPcYQDUV685F0D3Ce4e96435d9429903a5dFbEb5c")
    # {goal_obj} from {goal_find} to {goal_place}
    # target_name = goal[0].split(" from ")[1].split(" to ")[0].lower()
    target_name = goal[0]

    h , w = semantic_map.shape[0], semantic_map.shape[1]
    ratio = h/w if h>=w else w/h

    prompt = f"You are a professional spatial reasoner, assisting a navigation agent in finding a target object based on the top-down view semantic map."
    if w<=h:
        prompt += f"\nHere is a top-down view semantic map of an indoor environment. The lower left corner of the map is the coordinate origin (0.0, 0.0), the horizontal axis represents the X axis, the vertical axis represents the Y axis. The X axis range is (0.0, 10.0), the Y axis range is (0.0, {str(int(ratio*10.0))})."
    else:
        prompt += f"\nHere is a top-down view semantic map of an indoor environment. The lower left corner of the map is the coordinate origin (0.0, 0.0), the horizontal axis represents the X axis, the vertical axis represents the Y axis. The X axis range is (0.0, {str(int(ratio*10.0))}), the Y axis range is (0.0, 10.0)."
    prompt += f"\nIn the map, the brown square represents the agent's current location. The white area represents the unknown unexplored area. The green area represents the known navigatable area. The blue area represents historic areas navigated by the agent. The grey area represents the obstacles and objects in the room."
    
    prompt += f"\nPlease reason carefully and analyze step by step based on the semantic map and the above information. Answer the following questions one by one and give the accurate and detailed reasoning process:"

    prompt += f"\n1. Estimate the potential coordinates of target {target_name} on the map by observing the layout. Note: use the specific symbol <target>(x,y)</target> to answer this question where x and y are coordinates of {target_name}. The coordinates should be precise and in floating-point format."
    prompt += f"\n2. Consider the target coordinates that you indicate and agent's current location in brown, you must analyze carefully based on the layout, and then choose a actual-moving point. This actual-moving point cannot be too far from agent's current position. Through this point, agent can reach the potential target point you pointed out. Note: use the specific symbol <actual>(x,y)</actual> to answer this question."


    image = Image.fromarray(np.uint8(semantic_map.numpy()*255))
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

    messages = [
        {
            "role": "user", 
            "content": [
                {"type":"text", "text":prompt},
                {
                "type":"image_url",
                "image_url":{
                    "url":f"data:image/png;base64,{base64_image}",
                    "detail": "low"
                    }
                }
            ]
        }
        ]

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=messages
    )

    answer = completion.choices[0].message.content.strip().replace("\n"," ")

    if "<target>" in answer and "</target>" in answer:
        rel_coordinates = answer.split("<target>")[-1].split("</target>")[0]
        matches = re.findall(r"\(?(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)?", rel_coordinates)
        try:
            rel_coordinates = [float(matches[0][0]), float(matches[0][1])]
        except:
            rel_coordinates = []
    else:
        rel_coordinates = []

    if len(rel_coordinates)>0:

        if w<=h:
            coordinates = [(rel_coordinates[0]/10*w), (h-rel_coordinates[1]/(10*ratio)*h)]
        else:
            coordinates = [(rel_coordinates[0]/(10*ratio)*w), (h-rel_coordinates[1]/(10)*h)]
        
        abs_coordinates = [int(coordinates[0]), int(coordinates[1])]
    else:
        abs_coordinates = []


    if "<actual>" in answer and "</actual>" in answer:
        actual_rel_coordinates = answer.split("<actual>")[-1].split("</actual>")[0]
        matches = re.findall(r"\(?(-?\d+(?:\.\d+)?),\s*(-?\d+(?:\.\d+)?)\)?", actual_rel_coordinates)
        try:
            actual_rel_coordinates = [float(matches[0][0]), float(matches[0][1])]
        except:
            actual_rel_coordinates = []
    else:
        actual_rel_coordinates = []

    if len(actual_rel_coordinates)>0:

        if w<=h:
            actual_coordinates = [(actual_rel_coordinates[0]/10*w), (h-actual_rel_coordinates[1]/(10*ratio)*h)]
        else:
            actual_coordinates = [(actual_rel_coordinates[0]/(10*ratio)*w), (h-actual_rel_coordinates[1]/(10)*h)]
        
        # actual_abs_coordinates = [int(actual_coordinates[0]), int(actual_coordinates[1])]
        actual_abs_coordinates = [int(actual_coordinates[1]), int(actual_coordinates[0])]
    else:
        actual_abs_coordinates = []
    if len(actual_abs_coordinates) > 0:
        if actual_abs_coordinates[0] < seq_lmb[0, 0, 0]:
            actual_abs_coordinates[0] = seq_lmb[0, 0, 0]

        if actual_abs_coordinates[0] > seq_lmb[0, 0, 1]:
            actual_abs_coordinates[0] = seq_lmb[0, 0, 1] - 1

        if actual_abs_coordinates[1] < seq_lmb[0, 0, 2]:
            actual_abs_coordinates[1] = seq_lmb[0, 0, 2]

        if actual_abs_coordinates[1] > seq_lmb[0, 0, 3]:
            actual_abs_coordinates[1] = seq_lmb[0, 0, 3] - 1
    if len(actual_abs_coordinates) > 0:
        goal_map = torch.zeros_like(semantic_map[seq_lmb[0, 0, 0].item() : seq_lmb[0, 0, 1].item() , seq_lmb[0, 0, 2].item()  : seq_lmb[0, 0, 3].item() ])
        goal_map = goal_map[: , :, 0]
        goal_map [(actual_abs_coordinates[0] - seq_lmb[0, 0, 0]).item(), (actual_abs_coordinates[1] - seq_lmb[0, 0, 2]).item()] = 1.0
        goal_map = goal_map.unsqueeze(0)
        return goal_map
    else:
        return None