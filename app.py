import os
import base64
import math
#from app import app
from flask import flash, request, redirect, url_for, render_template,Flask
from werkzeug.utils import secure_filename

import io as io_norm
from skimage import io
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
from PIL import Image
import glob

from data_loader import RescaleT
from data_loader import ToTensor
from data_loader import ToTensorLab
from data_loader import SalObjDataset

from model import U2NET # full size version 173.6 MB
from model import U2NETP # small version u2net 4.7 MB

UPLOAD_FOLDER = "static/uploads"
FINAL_FOLDER = "static/final"

app = Flask(__name__)
app.secret_key = "secret key"


app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["FINAL_FOLDER"] = FINAL_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16*1024*1024

def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)

    dn = (d-mi)/(ma-mi)

    return dn

def convert_img(image_name,pred,d_dir, org_img_path):

    predict = pred
    predict = predict.squeeze()
    predict_np = predict.cpu().data.numpy()
    org_img = Image.open(org_img_path)

    im = Image.fromarray(predict_np*255).convert('RGB')
    img_name = image_name.split(os.sep)[-1]
    image = io.imread(image_name)
    imo = im.resize((image.shape[1],image.shape[0]),resample=Image.BILINEAR)
    # print(type(imo))
    pb_np = np.array(imo)

    aaa = img_name.split(".")
    bbb = aaa[0:-1]
    imidx = bbb[0]
    for i in range(1,len(bbb)):
        imidx = imidx + "." + bbb[i]

    
    
    cubw, cubh = 20,20
    wcubic = Image.new("RGB", (cubw, cubh), (255,255,255))
    gcubic = Image.new("RGB", (cubw, cubh), (238,238,238))

    whiteg = Image.new("RGB", imo.size, (255,255,255))
    blackg = Image.new("RGB", imo.size, (0,0,0))

    backg = Image.new("RGB", imo.size)
    width, height = imo.size
    for i in range(math.ceil(width/cubw)):
        for j in range(math.ceil(height/cubh)):
            if (i+j)%2:
                backg.paste(wcubic,(i*cubw, j*cubh))
            else:
                backg.paste(gcubic, (i*cubw, j*cubh))
    im1 = Image.Image.split(imo)
    alpha = im1[0]
    # checkerboard
    backg.paste(org_img, (0,0), alpha)
    # white
    whiteg.paste(org_img, (0,0), alpha)
    #black
    blackg.paste(org_img, (0,0), alpha)
    # transparent
    rgba = org_img.convert("RGBA")
    rgba.putalpha(alpha)

    # returns [original image, choose mode img]
    return [org_img, rgba, backg, whiteg, blackg]

def run_model():

    # --------- 1. get image path and name ---------
    model_name='u2net'

    image_dir = os.path.join(os.getcwd(), 'static', 'uploads')
    prediction_dir = os.path.join(os.getcwd(), 'static', "final" + os.sep)
    model_dir = os.path.join(os.getcwd(), 'saved_models', model_name, model_name + '.pth')

    img_name_list = glob.glob(image_dir + os.sep + '*')
    #print(img_name_list)

    # --------- 2. dataloader ---------
    #1. dataloader
    test_salobj_dataset = SalObjDataset(img_name_list = img_name_list,
                                        lbl_name_list = [],
                                        transform=transforms.Compose([RescaleT(320),
                                                                      ToTensorLab(flag=0)])
                                        )
    test_salobj_dataloader = DataLoader(test_salobj_dataset,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=1)

    # --------- 3. model define ---------
    if(model_name=='u2net'):
        print("...load U2NET---173.6 MB")
        net = U2NET(3,1)
    elif(model_name=='u2netp'):
        print("...load U2NEP---4.7 MB")
        net = U2NETP(3,1)

    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_dir))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()

    # --------- 4. inference for each image ---------
    final_dict = {}
    for i_test, data_test in enumerate(test_salobj_dataloader):

        print("inferencing:",img_name_list[i_test].split(os.sep)[-1])

        inputs_test = data_test['image']
        inputs_test = inputs_test.type(torch.FloatTensor)
        
        if torch.cuda.is_available():
            inputs_test = Variable(inputs_test.cuda())
        else:
            inputs_test = Variable(inputs_test)

        d1,d2,d3,d4,d5,d6,d7= net(inputs_test)

        # normalization
        pred = d1[:,0,:,:]
        pred = normPRED(pred)

        final_dict[img_name_list[i_test].split(os.sep)[-1]] = convert_img(img_name_list[i_test],pred,prediction_dir, img_name_list[i_test])

        del d1,d2,d3,d4,d5,d6,d7

    # returns dict mapping file name to [original image, nobg image, checker, whiteg, blackg]
    return final_dict


    

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    
@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
def upload_image():
    print(request.files)
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files["file"]
    file2 = request.files.getlist("file")
    print(file2)
    for file in file2:
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        else:
            flash('Allowed image types are -> png, jpg, jpeg, gif')
            return redirect(request.url)



    # dict mapping file name to [original image, nobg image, checker, whiteg, blackg]
    final_img_dict = run_model()

    image_dir = os.path.join(os.getcwd(), 'static', 'uploads')
    img_name_list = glob.glob(image_dir + os.sep + '*')
    for img in img_name_list:
        try:
            os.remove(img)
        except OSError as e:
            print("Error: %s:%s" % (img, e.strerror))
    base64_img_crop_list = {}
    base64_img_list = {}
    # [final, checker, white, black]
    for i in range(1,5):
        for final_img in final_img_dict:
            bytesio = io_norm.BytesIO()
            print(type(final_img_dict[final_img]))
            final_img_dict[final_img][i].save(bytesio, format = "PNG")
            bytesio.seek(0)
            base64_img = base64.b64encode(bytesio.getvalue())
            base64_img = base64_img.decode("ascii")
            base64_img_list[final_img]=[base64_img]

    # [original, final, checker, white, black]
    for i in range(5):
        for final_img in final_img_dict:
            bytesio = io_norm.BytesIO()
            resized_image = final_img_dict[final_img][i]
            resized_image.thumbnail((400,400))
            resized_image.save(bytesio, format = "PNG")
            bytesio.seek(0)
            base64_img = base64.b64encode(bytesio.getvalue())
            base64_img = base64_img.decode("ascii")
            if final_img in base64_img_crop_list:
                base64_img_crop_list[final_img].append(base64_img)
            else:
                base64_img_crop_list[final_img]=[base64_img]


    


    

    #print(base64_img)
    # print('upload_image filename: ' + filename)
    message = 'Image successfully uploaded and displayed below'
    # base64_img_list maps filename to [final, crop_orig, crop_final, crop_checker, crop_white, crop_black]
    # print(base64_img_list.values())
    print([x[-1][-1] for x in base64_img_list.values()])
    return render_template('upload.html', base64_img = base64_img_list, message = message, base64_img_crop_list=base64_img_crop_list)

    

@app.route('/display/<filename>')
def display_image(filename):
    print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='final/' + filename.rsplit('.',1)[0]+'.png'), code=301)

if __name__ == "__main__":
    app.run(debug=True)














