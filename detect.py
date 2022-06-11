from torchvision import transforms
from utils import *
from PIL import Image, ImageDraw, ImageFont
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import glob
from utils import tiff_to_numpy, resize
import torchvision.transforms.functional as FT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model checkpoint
checkpoint = 'BEST_checkpoint_ssd300.pth.tar'
checkpoint = torch.load(checkpoint)
start_epoch = checkpoint['epoch'] + 1
best_loss = checkpoint['best_loss']
print('\nLoaded checkpoint from epoch %d. Best loss so far is %.3f.\n' % (start_epoch, best_loss))
model = checkpoint['model'].module
model = model.to(device)
model.eval()

# Transforms
resize = transforms.Resize((300, 300))
#to_tensor = transforms.ToTensor()
#normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 #std=[0.229, 0.224, 0.225])
mean=[0.485, 0.456, 0.406]*32
std=[0.229, 0.224, 0.225]*32


def resize(image, dims=(300, 300), return_percent_coords=True):

    # image [96,h,w]
    # [1,96,h,w]才能resize
    image = image.unsqueeze(0)
    new_image = F.interpolate(image,size=dims)
    new_image = new_image.squeeze() # [96,h,w]
    # Resize bounding boxes
    old_dims = torch.FloatTensor([image.size()[3], image.size()[2], image.size()[3], image.size()[2]]).unsqueeze(0) # xyxy，所以取出whwh
    
    if not return_percent_coords:
        new_dims = torch.FloatTensor([dims[1], dims[0], dims[1], dims[0]]).unsqueeze(0)
        new_boxes = new_boxes * new_dims

    return new_image

def detect(tiff_image, rgb_image, min_score, max_overlap, top_k, suppress=None):
    """
    Detect objects in an image with a trained SSD300, and visualize the results.

    :param original_image: image, a PIL Image
    :param min_score: minimum threshold for a detected box to be considered a match for a certain class
    :param max_overlap: maximum overlap two boxes can have so that the one with the lower score is not suppressed via Non-Maximum Suppression (NMS)
    :param top_k: if there are a lot of resulting detection across all classes, keep only the top 'k'
    :param suppress: classes that you know for sure cannot be in the image or you do not want in the image, a list
    :return: annotated image, a PIL Image
    """

    # Transform
    #image = normalize(to_tensor(resize(original_image)))
    original_image = rgb_image
    image = torch.from_numpy(tiff_image).permute(2,0,1).float()/255. # 变为[h,w,9]的[0,1]的tensor
    image = resize(image, dims=(300, 300))
    image = FT.normalize(image, mean=mean, std=std)

    # Move to default device
    image = image.to(device)

    # Forward prop.
    predicted_locs, predicted_scores = model(image.unsqueeze(0))

    # Detect objects in SSD output
    det_boxes, det_labels, det_scores = model.detect_objects(predicted_locs, predicted_scores, min_score=min_score,
                                                             max_overlap=max_overlap, top_k=top_k)

    # Move detections to the CPU
    det_boxes = det_boxes[0].to('cpu')

    # Transform to original image dimensions
    original_dims = torch.FloatTensor(
        [original_image.width, original_image.height, original_image.width, original_image.height]).unsqueeze(0)
    det_boxes = det_boxes * original_dims

    # Decode class integer labels
    det_labels = [rev_label_map[l] for l in det_labels[0].to('cpu').tolist()]

    # If no objects found, the detected labels will be set to ['0.'], i.e. ['background'] in SSD300.detect_objects() in model.py
    if det_labels == ['background']:
        # Just return original image
        return original_image

    # Annotate
    annotated_image = original_image
    draw = ImageDraw.Draw(annotated_image)
    #font = ImageFont.truetype("./calibril.ttf", 15)
    font = ImageFont.load_default()

    # Suppress specific classes, if needed
    for i in range(det_boxes.size(0)):
        if suppress is not None:
            if det_labels[i] in suppress:
                continue

        # Boxes
        box_location = det_boxes[i].tolist()
        draw.rectangle(xy=box_location, outline=label_color_map[det_labels[i]])
        draw.rectangle(xy=[l + 1. for l in box_location], outline=label_color_map[
            det_labels[i]])  # a second rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 2. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a third rectangle at an offset of 1 pixel to increase line thickness
        # draw.rectangle(xy=[l + 3. for l in box_location], outline=label_color_map[
        #     det_labels[i]])  # a fourth rectangle at an offset of 1 pixel to increase line thickness

        # Text
        text_size = font.getsize(det_labels[i].upper())
        text_location = [box_location[0] + 2., box_location[1] - text_size[1]]
        textbox_location = [box_location[0], box_location[1] - text_size[1], box_location[0] + text_size[0] + 4.,
                            box_location[1]]
        draw.rectangle(xy=textbox_location, fill=label_color_map[det_labels[i]])
        draw.text(xy=text_location, text=det_labels[i].upper(), fill='white',
                  font=font)
    del draw

    return annotated_image


if __name__ == '__main__':
    test_txt_mode = True
    if test_txt_mode == True:
        with open('./TEST_images.json','r') as f:
            load_dict = json.load(f)
        for i in range(len(load_dict)):
            print(i)
            img_path = load_dict[i]
            filename_without_dir = os.path.basename(img_path)
            number = os.path.splitext(filename_without_dir)[0]
            number = number.replace('hy','rgb')
            original_image = tiff_to_numpy(img_path)
            rgb_image = Image.open('./VOC2007_Hyper/RGB/' + number + '.jpg', mode='r')
            result = detect(original_image, rgb_image, min_score=0.2, max_overlap=0.5, top_k=200)
            result.save('./results/' + number + '.jpg')
    else:
        file_list = sorted(glob.glob('./test/*'))
        for i in range(len(file_list)):
            print(i)
            img_path = file_list[i]
            filename_without_dir = os.path.basename(img_path)
            number = os.path.splitext(filename_without_dir)[0]
            original_image = tiff_to_numpy(img_path) # uint8 (426, 426, 9) numpy
            rgb_image = Image.open('./test_bmp/' + number + '.bmp', mode='r')
            result = detect(original_image, rgb_image, min_score=0.2, max_overlap=0.5, top_k=200)
            result.save('./results/' + number + '.jpg')
