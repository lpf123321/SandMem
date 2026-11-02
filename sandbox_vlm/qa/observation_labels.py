"""D054: label only actual recorded 2D observations on unchanged full frames."""
import hashlib
import math

LABEL_NOTICE=("\nImage labels Hn denote visual_id n in the memory. Boxes mark recorded observations. "
              "Different IDs may refer to the same physical object.")


def annotate_observations(image,links,*,frame_id,source_hw,watermark,font_path):
    from PIL import ImageDraw,ImageFont
    if type(frame_id) is not int or frame_id<0 or frame_id>watermark or image.mode!='RGB':
        raise ValueError('invalid source image or future frame')
    h,w=source_hw
    if min(h,w,image.height,image.width)<=0:raise ValueError('invalid image grid')
    result=image.copy();draw=ImageDraw.Draw(result);font=ImageFont.truetype(str(font_path),11)
    labels=[]
    for link in links:
        e=link['evidence'];gid=link['visual_instance_id']
        if (e['frame_id']!=frame_id or e['source_hw']!=list(source_hw)
                or not e['sam_mask_nonempty'] or type(gid) is not int or gid<0):
            raise ValueError('label lacks an actual same-frame observation')
        a,b,c,d=e['bbox_xyxy']
        if not all(math.isfinite(v) for v in (a,b,c,d)) or not(0<=a<c<=w and 0<=b<d<=h):
            raise ValueError('invalid actual source box')
        box=[a*image.width/w,b*image.height/h,c*image.width/w,d*image.height/h]
        draw.rectangle(box,outline=(255,40,30),width=1)
        label=f'H{gid}';x=max(0,min(int(box[0]),image.width-40));y=max(0,int(box[1])-13)
        draw.rectangle((x,y,x+draw.textlength(label,font=font)+3,y+12),fill='white')
        draw.text((x+1,y),label,fill=(180,0,0),font=font)
        labels.append(dict(visual_id=gid,frame_id=frame_id,source_bbox_xyxy=e['bbox_xyxy'],input_bbox_xyxy=box))
    return result,dict(labels=labels,source_pixel_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
        annotated_pixel_sha256=hashlib.sha256(result.tobytes()).hexdigest(),image_size_hw=[image.height,image.width])
