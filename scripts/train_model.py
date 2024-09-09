import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
import tensorflow as tf
import cv2
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
import segmentation_models as sm
import math
import glob
import albumentations as albu
from tensorflow.keras.callbacks import ModelCheckpoint

def get_segmentation_array(image_input, nClasses,
                           width, height, no_reshape=False, read_image_type=1):
    """ Load segmentation array from input """

    seg_labels = np.zeros((height, width, nClasses))

    if type(image_input) is np.ndarray:
        # It is already an array, use it as it is
        img = image_input
    elif isinstance(image_input, six.string_types):
        if not os.path.isfile(image_input):
            raise DataLoaderError("get_segmentation_array: "
                                  "path {0} doesn't exist".format(image_input))
        img = cv2.imread(image_input, read_image_type)
    else:
        raise DataLoaderError("get_segmentation_array: "
                              "Can't process input type {0}"
                              .format(str(type(image_input))))

    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_NEAREST)
    img = img[:, :, 0]

    for c in range(nClasses):
        seg_labels[:, :, c] = (img == c).astype(int)

    if not no_reshape:
        seg_labels = np.reshape(seg_labels, (width*height, nClasses))

    return seg_labels

class CustomDataGenerator(tf.keras.utils.Sequence):

    def __init__(self,
            images,
            masks,
            batch_size=16,
            num_classes=None,
            augmentation=None,
            preprocessing=None,
            img_dim=(320,320),
            ):
        self.images = images
        self.masks = masks
        self.train_len = len(images)
        self.batch_size = batch_size
        self.img_dim = img_dim
        self.mask_dim = img_dim
        self.num_classes = num_classes
        self.augmentation = augmentation
        self.preprocessing = preprocessing
        print(f"Found {len(self.images)} images belonging to {self.num_classes} classes")

    def __len__(self):
        ''' return total number of batches '''
        return math.ceil(self.train_len/self.batch_size)

    def __get_image_mask_pair(self, img_id, mask_id):
        img = cv2.imread(img_id)
        img = cv2.resize(img, self.img_dim, interpolation=cv2.INTER_NEAREST)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_id)
        _,mask = cv2.threshold(mask,100,255,cv2.THRESH_BINARY)
        mask[mask==255] = 1
        mask = cv2.resize(mask, self.mask_dim, interpolation=cv2.INTER_NEAREST_EXACT)
        mask = get_segmentation_array(mask, self.num_classes, self.mask_dim[0], self.mask_dim[1], no_reshape=True, read_image_type=1)
        mask = mask.astype('uint8')

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=img, mask=mask)
            img, mask = sample['image'], sample['mask']

        return img, mask

    def __getitem__(self, idx):
        x = []
        y = []
        batch_x = self.images[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.masks[idx * self.batch_size:(idx + 1) * self.batch_size]

        for img_id, mask_id in zip(batch_x, batch_y):
            image, label = self.__get_image_mask_pair(img_id, mask_id)
            x.append(image)
            y.append(label)
        return np.array(x), np.array(y)

def get_training_augmentation():
    train_transform = [
        albu.Downscale(always_apply=False, p=0.3, scale_min=0.75, scale_max=0.99, interpolation=3),
        albu.ShiftScaleRotate(always_apply=False, p=0.3, shift_limit_x=(-0.06, 0.06), shift_limit_y=(-0.06, 0.06), scale_limit=(-0.1, 0.1), rotate_limit=(-20, 20), interpolation=0, border_mode=4, value=(0, 0, 0), rotate_method='largest_box'),
        albu.MotionBlur(always_apply=False, p=0.2, blur_limit=3),
        albu.HorizontalFlip(always_apply=False, p=0.3),
        albu.RandomBrightnessContrast(always_apply=False, p=0.3, brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), brightness_by_max=True),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        albu.NoOp(p=1.0)
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.astype('float32')


def get_preprocessing():
    _transform = [
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def get_model(input_shape, backbone, weights="imagenet"):
    model = sm.Unet(backbone, classes=2, activation='softmax', 
                    input_shape=input_shape, encoder_weights=weights)
    model.compile(
    tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=sm.losses.categorical_focal_jaccard_loss,
    metrics=[sm.metrics.IOUScore(),
             ],
    )
    return model

def train_model(rgb_dir, gt_dir, model_path):
    rgb_images = glob.glob(rgb_dir+'*')
    gt_images = glob.glob(gt_dir+'*')
    
    X_train, X_val, y_train, y_val = train_test_split(rgb_images, gt_images, test_size=0.2, random_state=42)

    train_data = CustomDataGenerator(X_train, y_train, img_dim=(320,320),
    augmentation=get_training_augmentation(), 
    preprocessing=get_preprocessing(),
    num_classes=2,) 

    validation_data = CustomDataGenerator(X_val, y_val, img_dim=(320,320),
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(),
    num_classes=2,
    batch_size=1,)

    model = get_model(input_shape=(320,320,3), backbone='mobilenetv2')

    checkpoints_path = model_path
    checkpoints_callback = ModelCheckpoint(
                    filepath=checkpoints_path,
                    save_weights_only=True,
                    verbose=True,
                    save_best_only = True,
                    monitor="val_loss",
                )
    
    callbacks = [checkpoints_callback]

    epochs=100
    batch_size = 16
    
    history = model.fit(
               train_data,
               batch_size=batch_size,
               epochs=epochs,
               validation_data=validation_data,
               callbacks=callbacks,
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Treine o modelo de segmentação.')
    parser.add_argument('--rgb', required=True, help='Caminho para o diretório com imagens RGB')
    parser.add_argument('--groundtruth', required=True, help='Caminho para o diretório com imagens de verdade')
    parser.add_argument('--modelpath', required=True, help='Caminho para salvar o modelo treinado')
    
    args = parser.parse_args()
    
    train_model(args.rgb, args.groundtruth, args.modelpath)