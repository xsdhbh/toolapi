from paddleocr import PaddleOCR
import cv2
import numpy as np




def remove_large_image_watermark(image_byte, target_texts, output_path,
                                 max_size=2000, overlap=200,
                                 method='inpaint', confidence_th=0.5):
    """
    使用PaddleOCR识别指定文字并去除水印
    参数：
    image_byte: 输入图片字节
    target_texts: 需要识别的水印文字列表，如 ['水印1', '水印2']
    output_path: 输出图片路径
    method: 去水印方法，可选 'cover'（覆盖）或 'inpaint'（修复），默认为'inpaint'
    max_size: 单边最大尺寸，超过将进行分块处理
    overlap: 分块重叠像素
    """
    # 初始化OCR
    ocr = PaddleOCR(use_angle_cls=True, lang='ch')
    # 读取图片并获取尺寸
    img = bytes_to_ndarray(image_byte)
    if img is None:
        raise ValueError("无法读取图片文件")
    h, w = img.shape[:2]

    # 分块处理判断
    need_split = h > max_size or w > max_size

    if need_split:
        print(f"开始分块处理 ({w}x{h})...")
        # 生成分块坐标
        blocks = split_image_blocks(w, h, max_size, overlap)
        all_boxes = []

        # 并行处理分块（可改为多线程加速）
        for i, (x1, y1, x2, y2) in enumerate(blocks):
            block_img = img[y1:y2, x1:x2]
            block_result = ocr.ocr(block_img, cls=True)

            # 转换坐标到原图
            for line in block_result:
                for word_info in line:
                    box = word_info[0]
                    # 坐标转换
                    translated_box = [
                        [int(p[0] + x1), int(p[1] + y1)]
                        for p in box
                    ]
                    # 筛选条件：匹配任意一个目标文字
                    if (any(target in word_info[1][0] for target in target_texts) and
                            word_info[1][1] > confidence_th):
                        all_boxes.append(translated_box)
            print(f"已完成分块 {i + 1}/{len(blocks)}")
    else:
        # 直接处理
        result = ocr.ocr(img, cls=True)
        all_boxes = [
            [[int(p[0]), int(p[1])] for p in word_info[0]]
            for line in result
            for word_info in line
            if (any(target in word_info[1][0] for target in target_texts) and
                word_info[1][1] > confidence_th)
        ]

    # 后续处理（与之前相同）
    if not all_boxes:
        print("未检测到目标水印文字")
        return

    # 创建掩膜
    mask = create_mask_from_boxes(img.shape, all_boxes)

    # 去水印处理
    processed_img = remove_watermark(img, mask, method)

    # 保存结果
    cv2.imwrite(output_path, processed_img)
    print(f"处理完成：{output_path}")


def split_image_blocks(total_w, total_h, block_size, overlap):
    """生成分块坐标列表"""
    blocks = []
    for y in range(0, total_h, block_size - overlap):
        for x in range(0, total_w, block_size - overlap):
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(x + block_size, total_w)
            y2 = min(y + block_size, total_h)
            blocks.append((x1, y1, x2, y2))
    return blocks


def create_mask_from_boxes(img_shape, boxes, dilate=True):
    """根据文本框生成掩膜"""
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    for box in boxes:
        pts = np.array(box, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)

    if dilate:
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel)
    return mask


def remove_watermark(img, mask, method):
    """去水印核心方法"""
    if method == 'cover':
        img[mask == 255] = [255, 255, 255]
    elif method == 'inpaint':
        img = cv2.inpaint(img, mask, 5, cv2.INPAINT_TELEA)
    else:
        raise ValueError("Invalid method")
    return img


def bytes_to_ndarray(img_bytes: bytes):
    """字节转numpy数组

    Args:
        img_bytes (bytes): 图片字节

    Returns:
        _type_: _description_
    """
    image_array = np.frombuffer(img_bytes, dtype=np.uint8)
    image_np2 = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image_np2
