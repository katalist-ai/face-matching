import random

sexes = ["male", "female"]
# ages = ['child', 'adult']
ages = ["2 year old", "5 year old", "8 year old", "18 year old", "40 year old", "80 year old"]
children = {"2 year old", "5 year old", "8 year old"}
adults = {"18 year old", "40 year old", "80 year old"}
age_mapping = {
    "2 year old": "child",
    "5 year old": "child",
    "8 year old": "child",
    "18 year old": "adult",
    "40 year old": "adult",
    "80 year old": "adult",
}

ethnicities = ["caucasian", "asian", "black", "indian", "latino"]
camera_angles = ["shot from above", "shot from below", "straight on shot"]
shot_types = ["close shot", "medium shot", "wide shot"]


def get_aged_sex(age, sex):
    if sex == "male":
        if age in children:
            return "boy"
        elif age in adults:
            return "man"
        else:
            raise ValueError("Invalid age")
    elif sex == "female":
        if age in children:
            return "girl"
        elif age in adults:
            return "woman"
        else:
            raise ValueError("Invalid age")
    else:
        raise ValueError("Invalid sex")


def read_prompts(file_path):
    with open(file_path, "r") as file:
        data = file.read()
    return data.split("<>")


def add_keyword(prompt, keyword):
    return prompt + ", " + keyword


def construct_prompts(prompts):
    constructed_prompts = []
    for prompt in prompts:
        for sex in sexes:
            for age in ages:
                for ethnicity in ethnicities:
                    camera_angle = random.choice(camera_angles)
                    shot_type = random.choice(shot_types)
                    new_prompt = prompt.replace("PERSON", f"{age} {ethnicity} {get_aged_sex(age, sex)}")
                    new_prompt = add_keyword(new_prompt, camera_angle)
                    new_prompt = add_keyword(new_prompt, shot_type)
                    new_prompt = add_keyword(new_prompt, "high quality")
                    constructed_prompts.append(
                        (new_prompt, (age, ethnicity, sex, get_aged_sex(age, sex), camera_angle, shot_type))
                    )
    return constructed_prompts


def main():
    import os

    from utils.const import data_dir

    raw_prompts = read_prompts(os.path.join(data_dir, "prompts_raw.txt"))
    constructed_prompts = construct_prompts(raw_prompts)
    for prompt in constructed_prompts:
        print(prompt)
    print("NPROMPTS, ", len(constructed_prompts))


if __name__ == "__main__":
    main()
