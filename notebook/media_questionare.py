from PIL import Image
from myllm import VQA, ImageTextMatcher

sex_poses = [
    'kissing sex',
    'fondling sex',
    'handjob sex',
    'fingering sex',
    'titjob sex',
    'blowjob sex',
    'cunnilingus sex',
    'deepthroat sex',
    'doggy sex',
    'anal sex',
    'missionary sex',
    'cowgirl sex',
    'cumshot sex',
    'facial cumshot sex',
    '69 sex',
]

camera_horizontal_angle = [
    'Front (0 degree)',
    'Side Profile (90 Degrees)',
    'Back (180 Degrees)',
    'Slight Side Back (150 Degrees)',
    'Three-Quarter View (45 Degrees)',
    'Slight Side Angle (15-30 Degrees)',
]

camera_vertical_angle = [
    "eye level",
    "high angle",
    "low angle",
    "bird's eye view",
    "worm's eye view",
    "over the shoulder shot",
    "close up",
    "frog view",
]

class MediaQuestionare:
    def __init__(self):
        self.vqa = VQA()
        self.matcher = ImageTextMatcher()

        self.info_functions = {
            # vqa
            "explicit_content": lambda img: self.vqa.ask(img, "Does this photo contain explicit content?"),
            "cloth_color":      lambda img: self.vqa.ask(img, "which color or colors or nude does the girl wear?"),
            "pose":             lambda img: self.vqa.ask(img, "which pose does the girl perform?"),
            "pose_hand":        lambda img: self.vqa.ask(img, "what is the pose of her hand"),
            "pose_thigh":       lambda img: self.vqa.ask(img, "what is the pose of her thigh"),
            "pose_crotch":      lambda img: self.vqa.ask(img, "what is the pose of her crotch"),
            "pose_leg":         lambda img: self.vqa.ask(img, "what is the pose of her leg"),

            # # vqa ranking
            # 'vertival_angle':   lambda img: self.vqa.rank(img, 'in which angle is this picture filmed?', camera_vertical_angle),
            # 'horizontal_angle': lambda img: self.vqa.rank(img, 'in which angle is this picture filmed?', camera_horizontal_angle),
            # 'sex':              lambda img: self.vqa.rank(img, 'which sex pose is contained in this picture.', sex_poses),
        
            # text matching
            'is_selfie':        lambda img: self.matcher.text_match(img, 'This is a selfie.'),

            # **{'vertical angle ' + angle: lambda img, x=angle: self.matcher.text_match(img, f'In terms of filming angle, this picture is in a {x} view')
               # for angle in camera_vertical_angle},
            # **{'horizontal angle ' + angle: lambda img, x=angle: self.matcher.text_match(img, f'In terms of filming angle, this picture is in a {x} view')
               # for angle in camera_horizontal_angle},
            # **{'Sex ' + pose: lambda img, x=pose: self.matcher.text_match(img, f'{x}')
               # for pose in sex_poses},
            
        }

    def get_image_size(self, image):
        """Returns the size of the image."""
        return image.size

    def get_image_format(self, image):
        """Returns the format of the image."""
        return image.format

    def process_image(self, image):
        """Processes the given PIL image and returns a dictionary of information."""
        if not isinstance(image, Image.Image):
            raise ValueError("The provided input is not a valid PIL Image.")

        info_dict = {}
        for key, func in self.info_functions.items():
            res = func(image)
            info_dict[key] = res

        return info_dict
