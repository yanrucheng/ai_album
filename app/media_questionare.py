from PIL import Image
from my_llm import VQA, ImageTextMatcher
from typing import Dict, Callable
from media_utils import MediaOperator

sex_poses = [ # accuracy low warning
    'kissing',
    'fondling',
    'handjob',
    'fingering sex',
    'titjob',
    'blowjob',
    'cunnilingus',
    'deepthroat',
    'doggy style',
    'anal sex',
    'missionary',
    'cowgirl',
    'cumshot',
    'facial cumshot',
    '69 sex',
]

camera_horizontal_angle = [ # accuracy low warning
    'Front (0 degree)',
    'Side Profile (90 Degrees)',
    'Back (180 Degrees)',
    'Slight Side Back (150 Degrees)',
    'Three-Quarter View (45 Degrees)',
    'Slight Side Angle (15-30 Degrees)',
]

camera_vertical_angle = [ # accuracy low warning
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

    def is_rotated_old(self, img):
        descriptions = {
            0: 'no rotation, standard orient',
            90: 'photo rotated 90° to right',
            180: 'this photo is upside-down',
            270: 'photo rotated 90° to left',
        }

        # Evaluate the confidence for each rotation
        confidences = {angle: self.matcher.text_match(img, desc) for angle, desc in descriptions.items()}

        # Find the rotation with the highest confidence
        highest_confidence_angle = max(confidences.keys(), key=confidences.get)
        highest_confidence = max(confidences.values()) / sum(confidences.values())
        result = {
            'rotate_0': confidences[0],
            'rotate_90': confidences[90],
            'rotate_180': confidences[180],
            'rotate_270': confidences[270],
            'rotate': highest_confidence_angle,
            'rotate_confidence': highest_confidence,
        }

        return result

    def is_rotated(self, img):
        '''This is sigificantly more accurate than is_rotated_old.
        Because text-matcher handles 'up-side down' most accurately.
        This function rotate the image and keep asking whether is upside-down.
        '''
        confidences = {0:0, 90:0, 180:0, 270:0}
        for a in (180, 0, 90, 270):
            img_ = MediaOperator.rotate_image(img, a)
            conf = self.matcher.text_match(img_, 'this photo is upside-down')
            confidences[(180 - a) % 360] = conf
            if conf > 0.5: # early ending threshold
                break # early pruning for fast inference

        # Find the rotation with the highest confidence
        highest_confidence_angle = max(confidences.keys(), key=confidences.get)
        highest_confidence = max(confidences.values()) / sum(confidences.values())
        result = {
            'rotate_0': confidences[0],
            'rotate_90': confidences[90],
            'rotate_180': confidences[180],
            'rotate_270': confidences[270],
            'rotate': highest_confidence_angle,
            'rotate_confidence': highest_confidence,
        }
        return result


    def is_selfie(self, img):
        '''When confidence > 0.25. Could be marked as selfie.'''
        qs = {
            'is_selfie': lambda img: self.matcher.text_match(img, 'This is a selfie.'),
        }
        return self._process_image(img)

    def sex_pose(self, img):
        '''Not recommended. low accuracy'''
        qs = {
            'sex': lambda img: self.vqa.rank(img, 'which sex pose is contained in this picture.', sex_poses),
        }
        return self._process_image(img)

    def sex_pose_detail(self, img):
        '''Not recommended. low accuracy'''
        qs = {
            **{'Sex ' + pose: lambda img, x=pose: self.matcher.text_match(img, f'{x}')
                for pose in sex_poses},
        }
        return self._process_image(img)


    def film_angle(self, img):
        '''Not recommended. low accuracy'''
        qs = {
            'vertival_angle':   lambda img: self.vqa.rank(img, 'in which angle is this picture filmed?', camera_vertical_angle),
            'horizontal_angle': lambda img: self.vqa.rank(img, 'in which angle is this picture filmed?', camera_horizontal_angle),
        }
        return self._process_image(img)

    def film_angle_detail(self, img):
        '''Not recommended. low accuracy'''
        qs = {
            **{'vertical angle ' + angle: lambda img, x=angle: self.matcher.text_match(img, f'In terms of filming angle, this picture is in a {x} view')
               for angle in camera_vertical_angle},
            **{'horizontal angle ' + angle: lambda img, x=angle: self.matcher.text_match(img, f'In terms of filming angle, this picture is in a {x} view')
               for angle in camera_horizontal_angle},
        }
        return self._process_image(img)

    def is_explicit(self, img):
        qs = { "explicit_content": lambda img: self.vqa.ask(img, "Does this photo contain explicit content?"), }
        return self._process_image(img)

    def cloth_color(self, img):
        qs = { "cloth_color": lambda img: self.vqa.ask(img, "which color or colors or nude does the girl wear?"), }
        return self._process_image(img)

    def pose(self, img):
        qs = {
            "pose":             lambda img: self.vqa.ask(img, "which pose does the girl perform?"),
            "pose_hand":        lambda img: self.vqa.ask(img, "what is the pose of her hand"),
            "pose_thigh":       lambda img: self.vqa.ask(img, "what is the pose of her thigh"),
            "pose_crotch":      lambda img: self.vqa.ask(img, "what is the pose of her crotch"),
            "pose_leg":         lambda img: self.vqa.ask(img, "what is the pose of her leg"),
        }
        return self._process_image(img)


    def _process_image(self, image, question_dict: Dict[str, Callable[[str], str]]):
        """Processes the given PIL image and returns a dictionary of information."""
        if not isinstance(image, Image.Image):
            raise ValueError("The provided input is not a valid PIL Image.")

        info_dict = {}
        for key, func in question_dict.items():
            res = func(image)
            info_dict[key] = res

        return info_dict

