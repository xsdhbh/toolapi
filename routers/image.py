from datetime import datetime
from typing import List

from fastapi import APIRouter, UploadFile

import utils.image
from models.ResponseModel import ResponseModel
from utils.common import *

router = APIRouter(prefix="/image", tags=["图片去水印、orc"])

@router.post(path="/watermark", summary="去水印", response_model=ResponseModel)
def watermark(file: UploadFile, target_texts: List[str]):
    response: ResponseModel = ResponseModel()
    if file.filename.endswith((".jpg", ".png", ".jpeg")):
        file_data = file.file
        file_bytes = file_data.read()
        now = datetime.now()
        year = now.strftime("%Y")
        month = now.strftime("%m")

        # 创建年份和月份目录
        year_dir = os.path.join(BASE_DIR, year)
        month_dir = os.path.join(year_dir, month)

        # 确保目录存在
        ensure_directory_exists(month_dir)

        # 生成随机文件名
        random_filename = generate_random_filename(file.filename)
        file_path = os.path.join(month_dir, random_filename)

        utils.image.remove_large_image_watermark(
            image_byte=file_bytes,
            target_texts=target_texts,
            output_path=file_path,
            max_size=2000,  # 单边最大尺寸
            overlap=300,  # 分块重叠像素
            method='inpaint'
        )

    else:
        response.code = 0
        response.msg = "请上传 .jpg 或 .png 格式图片"

    return response
