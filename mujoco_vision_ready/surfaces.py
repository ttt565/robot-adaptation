"""Independent appearance and actual contact properties, plus an RGB-only camera."""

import io
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

import engine
from appearance import KINDS, make_texture


CASES = ["normal", "visible_surface_only", "hidden_low_friction", "visible_low_friction", "motor_fault"]


def scene_xml(texture_specs, before, after, friction_before=.8, friction_after=.8):
    xml = ET.fromstring((Path(__file__).parent / "robot.xml").read_text(encoding="utf-8"))
    world, asset = xml.find("worldbody"), xml.find("asset")
    floor = world.find("geom[@name='floor']")
    floor.set("pos", "0 0 -0.2")  # 底部兜底地面不与上方路面同时接触。
    assets = {}
    material_names = []
    for index, (kind, seed) in enumerate(texture_specs):
        name = f"surface_{index}"
        file_name = name + ".png"
        output = io.BytesIO()
        Image.fromarray(make_texture(kind, seed)).save(output, format="PNG")
        assets[file_name] = output.getvalue()
        ET.SubElement(asset, "texture", name=name, type="2d", file=file_name)
        # texuniform使每米重复尺度相同，两个不同长度地块不会改变纹理尺度。
        ET.SubElement(asset, "material", name=name + "_material", texture=name,
                      texuniform="true", texrepeat="2 2", reflectance="0")
        material_names.append(name + "_material")
    contact = ET.SubElement(xml, "contact")
    for index, (center, half_length, material, mu) in enumerate([
            (-1., 5., material_names[before], friction_before),
            (14., 10., material_names[after], friction_after)]):
        name = f"terrain_{index}"
        ET.SubElement(world, "geom", name=name, type="box", pos=f"{center} 0 -0.05",
                      size=f"{half_length} 4 0.05", material=material, group="0", priority="0")
        for tire in ["left_tire", "right_tire"]:
            # 显式pair避免轮胎较高priority盖过地面摩擦；支撑球仍保持自身低摩擦。
            ET.SubElement(contact, "pair", geom1=tire, geom2=name, condim="3",
                          solref="0.008 1", solimp="0.95 0.99 0.001",
                          friction=f"{mu} {mu} 0.001 0.0001 0.0001")
    chassis = world.find("body[@name='chassis']")
    ET.SubElement(chassis, "camera", name="terrain_camera", pos="0.24 0 0.23",
                  xyaxes="0 -1 0 0.70710678 0 0.70710678", fovy="60")
    # 识别RGB中不包含机器人自身、审计着色、参考球或文字。
    for geom in chassis.iter("geom"):
        geom.set("group", "1")
    return ET.tostring(xml, encoding="unicode"), assets, material_names


class SurfacePlant(engine.ContactPlant):
    def __init__(self, config, case, texture_seed=1000):
        self.case = case
        self.visible_change = case in ["visible_surface_only", "visible_low_friction"]
        self.mu_change = case in ["hidden_low_friction", "visible_low_friction"]
        after_kind = "speckled" if self.visible_change else "tile_grid"
        specs = [("tile_grid", texture_seed), (after_kind, texture_seed + 1 if self.visible_change else texture_seed)]
        xml, assets, _ = scene_xml(specs, 0, 1, .8, config["vision_low_friction"] if self.mu_change else .8)
        scenario = "left_motor_drop" if case == "motor_fault" else "none"
        super().__init__(config, scenario, xml, assets)
        self.terrain_ids = [self.model.geom(f"terrain_{index}").id for index in [0, 1]]

    def audit(self):
        audit = super().audit()
        mu, contacted = [], []
        for contact in self.data.contact:
            pair = [int(contact.geom1), int(contact.geom2)]
            if any(geom in self.tires for geom in pair):
                for index, terrain in enumerate(self.terrain_ids):
                    if terrain in pair:
                        mu.append(contact.friction[0])
                        contacted.append(index)
        audit["contact_mu"] = float(np.mean(mu)) if mu else np.nan
        audit["after_surface_contact"] = bool(1 in contacted)
        return audit


class GroundCamera:
    def __init__(self, plant):
        self.plant = plant
        self.renderer = mujoco.Renderer(plant.model, height=240, width=320)
        self.options = mujoco.MjvOption()
        self.options.geomgroup[1] = 0
        self.options.geomgroup[5] = 0

    def read_rgb(self):
        self.renderer.update_scene(self.plant.data, camera="terrain_camera", scene_option=self.options)
        return self.renderer.render().copy()

    def close(self):
        self.renderer.close()


def collect_dataset(output_dir):
    """每个样本由独立纹理seed生成；训练/测试不共享纹理，不拆连续视频。"""
    output_dir = Path(output_dir)
    specs, split, labels = [], [], []
    for kind in KINDS:
        for section, seeds in [("train", range(20)), ("test", range(200, 210))]:
            for seed in seeds:
                specs.append((kind, seed))
                split.append(section)
                labels.append(kind)
    xml, assets, materials = scene_xml(specs, 0, 0)
    config = __import__("json").loads((Path(__file__).parent / "config.json").read_text())
    plant = engine.ContactPlant(config, "none", xml, assets)
    camera = GroundCamera(plant)
    ids = [plant.model.geom(f"terrain_{index}").id for index in [0, 1]]
    rng = np.random.default_rng(712)
    images = []
    try:
        for index, material in enumerate(materials):
            plant.model.geom_matid[ids] = plant.model.mat(material).id
            x, y = rng.uniform(-.7, .7, 2)
            angle = rng.uniform(-.35, .35)
            plant.data.qpos[:3] = [x, y, .1]
            plant.data.qpos[3:7] = [math.cos(angle / 2), 0, 0, math.sin(angle / 2)]
            light = rng.uniform(.6, 1.0)
            plant.model.light_diffuse[:] = light
            plant.model.light_ambient[:] = rng.uniform(.15, .4)
            mujoco.mj_forward(plant.model, plant.data)
            rgb = camera.read_rgb()
            images.append(rgb)
            if index % 10 == 0:
                print(f"Rendered classifier dataset {index+1}/{len(materials)}", flush=True)
    finally:
        camera.close()
    np.savez_compressed(output_dir / "appearance_dataset.npz", images=np.array(images),
                        labels=np.array(labels), split=np.array(split), texture_seeds=np.array([s for _, s in specs]))
    return images, labels, split
