
import sys
from pathlib import Path
import logging
from utils.tools import *
from superpoint.superpoint import SuperPoint
import matplotlib.pyplot as plt


class SuperPointDetector(object):
    default_config = {
        "descriptor_dim": 256,
        "nms_radius": 4,
        "keypoint_threshold": 0.005,
        "max_keypoints": -1,
        "remove_borders": 4,
        "path": Path(__file__).parent / "superpoint/superpoint_v1.pth",
        "cuda": True
    }

    def __init__(self, config={}):
        # 初始化函数，接收一个config参数，默认为空字典
        self.config = self.default_config
        # 将默认配置赋值给self.config
        self.config = {**self.config, **config}
        # 将传入的config参数与默认配置合并，并赋值给self.config
        logging.info("SuperPoint detector config: ")
        logging.info(self.config)

        # 打印SuperPoint检测器的配置信息
        self.device = 'cuda' if torch.cuda.is_available() and self.config["cuda"] else 'cpu'

        logging.info("creating SuperPoint detector...")
        self.superpoint = SuperPoint(self.config).to(self.device)

    def __call__(self, image):
        if image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        logging.debug("detecting keypoints with superpoint...")
        image_tensor = image2tensor(image, self.device)
        pred = self.superpoint({'image': image_tensor})

        ret_dict = {
            "image_size": np.array([image.shape[0], image.shape[1]]),
            "torch": pred,
            "keypoints": pred["keypoints"][0].cpu().detach().numpy(),
            "scores": pred["scores"][0].cpu().detach().numpy(),
            "descriptors": pred["descriptors"][0].cpu().detach().numpy().transpose()
        }

        return ret_dict


if __name__ == "__main__":
    img = cv2.imread(r"Redal/datasets_vg/datasets/nwpu/test/queries/small_19_23.tif")
    plt.imshow(img)
    plt.show()
    detector = SuperPointDetector()
    print(detector)
    kptdescs = detector(img)

    img = plot_keypoints(img, kptdescs["keypoints"], kptdescs["scores"])
    plt.imshow(img)
    plt.show()
    cv2.imwrite('Redal/img.jpg', img)