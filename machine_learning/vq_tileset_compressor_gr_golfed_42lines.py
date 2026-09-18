#!/home/rich/MyCoding/venvMyCoding/bin/python
import gradio as g
import numpy as n
from PIL import Image as I,ImageDraw as D
from sklearn.cluster import KMeans as K
def zoom(a,f=2):return n.kron(a,n.ones((f,f,1)))
def image_to_blocks(im,block_size):
 h,w,_=im.shape;B=block_size;h,w=h//B,w//B;im=im[:h*B,:w*B];return im.reshape(h,B,w,B,3).transpose(0,2,1,3,4).reshape(-1,B*B*3),h,w,im
def vectors_to_pixels(x,h,w,block_size):
 B=block_size;return x.reshape(h,w,B,B,3).transpose(0,2,1,3,4).reshape(h*B,w*B,3)
def build_codebook_preview(codebook,num_codes,block_size):
 c=min(20,num_codes)
 while num_codes%c and c>1:c-=1
 return zoom(vectors_to_pixels(codebook[n.argsort(n.median(codebook[:,::3],1))],num_codes//c,c,block_size),3)
def ycbcr_to_rgb_array(a):return n.asarray(I.fromarray(n.clip(a,0,255).astype(n.uint8),mode='YCbCr').convert('RGB'))
def find_representative_blocks(a):return n.argmin(a,axis=0)
def draw_representative_tiles(a,b,bw,block_size):
 d=D.Draw(I.fromarray(a).convert('RGB'));w=max(1,block_size//8)
 for i in b:
  r,c=divmod(int(i),bw);x,y=c*block_size,r*block_size;d.rectangle((x,y,x+block_size-1,y+block_size-1),outline=(255,0,0),width=w)
 return d._image
def compress_image(input_image,block_size,num_codes,luma_weight,show_debug_tiles,progress=g.Progress()):
    if input_image is None:
        return None,None,"Please load an image first.",g.update(value=None,visible=False)
    block_size,num_codes,luma_weight=int(block_size),int(num_codes),float(luma_weight)
    im=n.asarray(input_image.convert('YCbCr')).astype(n.float32);h,w,_=im.shape
    if h < block_size or w < block_size:
        return None,None,f'Image is smaller than the block size ({block_size}x{block_size}). Choose a smaller block size or a larger image.',g.update(value=None,visible=False)
    X,bh,bw,cropped_im=image_to_blocks(im,block_size);num_codes=min(num_codes,len(X));Y=X.copy();Y[:,::3]*=luma_weight;k=K(n_clusters=num_codes,random_state=0,n_init='auto').fit(Y);codes=k.predict(Y);b=find_representative_blocks(k.transform(Y));codebook=X[b];result=ycbcr_to_rgb_array(vectors_to_pixels(codebook[codes],bh,bw,block_size));codebook_preview=ycbcr_to_rgb_array(build_codebook_preview(codebook,num_codes,block_size));debug_image=None
    if show_debug_tiles:
        debug_image=draw_representative_tiles(ycbcr_to_rgb_array(cropped_im),b,bw,block_size)
    summary=f'Orig image cropped to: {cropped_im.shape[1]} x {cropped_im.shape[0]} pixels\nImage size in blocks: {bw} x {bh} ({len(X)} blocks total)\nBlock size: {block_size} x {block_size} pixels\nno of unique tiles: {num_codes}\n'
    return I.fromarray(result),I.fromarray(codebook_preview),summary,g.update(value=debug_image,visible=show_debug_tiles)
with g.Blocks(title='Tileset img Compressor',theme=g.themes.Soft()) as demo:
 with g.Row():
  with g.Column():
   input_image=g.Image(label='Input',type='pil',height=800);block_size=g.Slider(minimum=2,maximum=64,value=16,step=2,label='tile size (w/h in px)');num_codes=g.Slider(minimum=2,maximum=512,value=180,step=1,label='no. of unique tiles)');luma_weight=g.Slider(minimum=.1,maximum=5,value=1.5,step=.1,label='luma weight (higher = prioritize brightness over color)');show_debug_tiles=g.Checkbox(label='Debug: outline representative tiles on original image',value=False);compress_btn=g.Button('Compress Image',variant='primary')
  with g.Column():output_image=g.Image(label='Decoded',type='pil',height=800);summary_text=g.Textbox(label='Summary',lines=6,interactive=False)
 with g.Row():codebook_image=g.Image(label='Tileset, ordered by brightness');debug_image=g.Image(label='Representative tiles (debug)',visible=False)
 show_debug_tiles.change(lambda v:g.update(visible=v),show_debug_tiles,debug_image);compress_btn.click(compress_image,[input_image,block_size,num_codes,luma_weight,show_debug_tiles],[output_image,codebook_image,summary_text,debug_image])
if __name__ == "__main__":
 demo.launch(inbrowser=True)