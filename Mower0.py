from __future__ import annotations
import copy
import csv
import ctypes
import colorlog
import cv2
import hashlib
import hmac
import inspect
import json
import logging
import os
import pystray
import pathlib
import requests
import smtplib
import sys
import threading
import time
import warnings
import yaml
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
from typing import Optional
from urllib import parse
from arknights_mower.data import agent_list
from arknights_mower.utils import (character_recognize, config, detector, segment)
from arknights_mower.utils import typealias as tp
from arknights_mower.utils.datetime import the_same_time
from arknights_mower.utils.device.adb_client import ADBClient
from arknights_mower.utils.device.minitouch import MiniTouch
from arknights_mower.utils.device.scrcpy import Scrcpy
from arknights_mower.utils.email import maa_template
from arknights_mower.utils.digit_reader import DigitReader
from arknights_mower.utils.operators import Operator, Operators
from arknights_mower.utils.pipe import push_operators
from arknights_mower.utils.scheduler_task import SchedulerTask
from arknights_mower.utils.solver import BaseSolver
from arknights_mower.utils.recognize import Recognizer, RecognizeError
from ctypes import CFUNCTYPE, c_int, c_char_p, c_void_p
from logging.handlers import RotatingFileHandler
import ttkbootstrap as 界面
from ttkbootstrap.constants import *
from ttkbootstrap.scrolled import ScrolledFrame
from tkinter import *
from pathlib import Path
from PIL import Image, ImageTk
from pystray import MenuItem, Menu


def warn(*args, **kwargs): pass


warnings.warn = warn

from paddleocr import PaddleOCR
from arknights_mower.strategy import Solver


class 设备控制(object):
    class 操控(object):
        def __init__(self, device: 设备控制, client: ADBClient = None, touch_device: str = None) -> None:
            self.device = device
            self.minitouch = None
            self.scrcpy = None

            if config.ADB_CONTROL_CLIENT == 'minitouch': self.minitouch = MiniTouch(client, touch_device)
            elif config.ADB_CONTROL_CLIENT == 'scrcpy': self.scrcpy = Scrcpy(client)
            else:
                # MiniTouch does not support Android 10+
                if int(client.android_version().split('.')[0]) < 10: self.minitouch = MiniTouch(client, touch_device)
                else: self.scrcpy = Scrcpy(client)

        def tap(self, point: tuple[int, int]) -> None:
            if self.minitouch: self.minitouch.tap([point], self.device.display_frames())
            elif self.scrcpy: self.scrcpy.tap(point[0], point[1])
            else: raise NotImplementedError

        def swipe(self, start: tuple[int, int], end: tuple[int, int], duration: int) -> None:
            if self.minitouch: self.minitouch.swipe([start, end], self.device.display_frames(), duration=duration)
            elif self.scrcpy: self.scrcpy.swipe(start[0], start[1], end[0], end[1], duration / 1000)
            else: raise NotImplementedError

        def swipe_ext(self, points: list[tuple[int, int]], durations: list[int], up_wait: int) -> None:
            if self.minitouch:
                self.minitouch.swipe(points, self.device.display_frames(), duration=durations, up_wait=up_wait)
            elif self.scrcpy:
                total = len(durations)
                for idx, (S, E, D) in enumerate(zip(points[:-1], points[1:], durations)):
                    self.scrcpy.swipe(S[0], S[1], E[0], E[1], D / 1000, up_wait / 1000 if idx == total - 1 else 0,
                                      fall=idx == 0, lift=idx == total - 1)
            else: raise NotImplementedError

    def __init__(self, device_id: str = None, connect: str = None, touch_device: str = None) -> None:
        self.device_id = device_id
        self.connect = connect
        self.touch_device = touch_device
        self.client = None
        self.control = None
        self.start()

    def start(self) -> None:
        self.client = ADBClient(self.device_id, self.connect)
        self.control = 设备控制.操控(self, self.client)

    def run(self, cmd: str) -> Optional[bytes]:
        return self.client.run(cmd)

    def launch(self, app: str) -> None:
        """ launch the application """
        self.run(f'am start -n {app}')

    def exit(self, app: str) -> None:
        """ exit the application """
        self.run(f'am force-stop {app}')

    def send_keyevent(self, keycode: int) -> None:
        """ send a key event """
        logger.debug(f'keyevent: {keycode}')
        command = f'input keyevent {keycode}'
        self.run(command)

    def send_text(self, text: str) -> None:
        """ send a text """
        logger.debug(f'text: {repr(text)}')
        text = text.replace('"', '\\"')
        command = f'input text "{text}"'
        self.run(command)

    def screencap(self, save: bool = False) -> bytes:
        """ get a screencap """
        command = 'screencap -p 2>/dev/null'
        screencap = self.run(command)
        if save: save_screenshot(screencap)
        return screencap

    def current_focus(self) -> str:
        """ detect current focus app """
        command = 'dumpsys window | grep mCurrentFocus'
        line = self.run(command).decode('utf8')
        return line.strip()[:-1].split(' ')[-1]

    def display_frames(self) -> tuple[int, int, int]:
        """ get display frames if in compatibility mode"""
        if not config.MNT_COMPATIBILITY_MODE: return None

        command = 'dumpsys window | grep DisplayFrames'
        line = self.run(command).decode('utf8')
        """ eg. DisplayFrames w=1920 h=1080 r=3 """
        res = line.strip().replace('=', ' ').split(' ')
        return int(res[2]), int(res[4]), int(res[6])

    def tap(self, point: tuple[int, int]) -> None:
        """ tap """
        logger.debug(f'tap: {point}')
        self.control.tap(point)

    def swipe(self, start: tuple[int, int], end: tuple[int, int], duration: int = 100) -> None:
        """ swipe """
        logger.debug(f'swipe: {start} -> {end}, duration={duration}')
        self.control.swipe(start, end, duration)

    def swipe_ext(self, points: list[tuple[int, int]], durations: list[int], up_wait: int = 500) -> None:
        """ swipe_ext """
        logger.debug(f'swipe_ext: points={points}, durations={durations}, up_wait={up_wait}')
        self.control.swipe_ext(points, durations, up_wait)

    def check_current_focus(self):
        """ check if the application is in the foreground """
        if not self.current_focus() == f"{config.APPNAME}/{config.APP_ACTIVITY_NAME}":
            self.launch(f"{config.APPNAME}/{config.APP_ACTIVITY_NAME}")
            # wait for app to finish launching
            time.sleep(10)


class 干员排序方式(Enum):
    工作状态 = 1
    技能 = 2
    心情 = 3
    信赖值 = 4


干员排序方式位置 = {
    干员排序方式.工作状态: (1560 / 2496, 96 / 1404),
    干员排序方式.技能: (1720 / 2496, 96 / 1404),
    干员排序方式.心情: (1880 / 2496, 96 / 1404),
    干员排序方式.信赖值: (2050 / 2496, 96 / 1404),
}

BASIC_FORMAT = '%(asctime)s - %(levelname)s - %(relativepath)s:%(lineno)d - %(funcName)s - %(message)s'
DATE_FORMAT = '%m-%d %H:%M:%S'
basic_formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)


class PackagePathFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        pathname = record.pathname
        record.relativepath = None
        abs_sys_paths = map(os.path.abspath, sys.path)
        for path in sorted(abs_sys_paths, key=len, reverse=True):  # longer paths first
            if not path.endswith(os.sep):
                path += os.sep
            if pathname.startswith(path):
                record.relativepath = os.path.relpath(pathname, path)
                break
        return True


class MaxFilter(object):
    def __init__(self, max_level: int) -> None:
        self.max_level = max_level

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno <= self.max_level:
            return True


class Handler(logging.StreamHandler):
    def __init__(self, pipe):
        logging.StreamHandler.__init__(self)
        self.pipe = pipe
        self.pipe.tag_configure("TIME", foreground="LightSlateBlue")
        self.pipe.tag_configure("INFO", foreground="ForestGreen")
        self.pipe.tag_configure("WARNING", foreground="Goldenrod")
        self.pipe.tag_configure("ERROR", foreground="red")
        self.pipe.tag_configure("DEBUG", foreground="gray")

    def emit(self, record):
        信息列表 = self.format(record).split(' - ')
        self.pipe.configure(state='normal')
        self.pipe.insert(END, 信息列表[0] + " ", "TIME")
        self.pipe.insert(END, 信息列表[-1] + '\n', record.levelname)
        self.pipe.yview(END)    # 自动滚动到底部


chlr = logging.StreamHandler(stream=sys.stdout)
chlr.setFormatter(basic_formatter)
chlr.setLevel('INFO')
chlr.addFilter(MaxFilter(logging.INFO))
chlr.addFilter(PackagePathFilter())

ehlr = logging.StreamHandler(stream=sys.stderr)
ehlr.setFormatter(basic_formatter)
ehlr.setLevel('WARNING')
ehlr.addFilter(PackagePathFilter())

logger = logging.getLogger(__name__)
logger.setLevel('DEBUG')
logger.addHandler(chlr)
logger.addHandler(ehlr)


def init_fhlr(pipe=None) -> None:
    """ initialize log file """
    if config.LOGFILE_PATH is None:
        return
    folder = Path(config.LOGFILE_PATH)
    folder.mkdir(exist_ok=True, parents=True)
    fhlr = RotatingFileHandler(
        folder.joinpath('runtime.log'),
        encoding='utf8',
        maxBytes=10 * 1024 * 1024,
        backupCount=config.LOGFILE_AMOUNT,
    )
    fhlr.setFormatter(basic_formatter)
    fhlr.setLevel('DEBUG')
    fhlr.addFilter(PackagePathFilter())
    logger.addHandler(fhlr)
    if pipe is not None:
        wh = Handler(pipe)
        wh.setLevel(logging.INFO)
        logger.addHandler(wh)
        wh.setFormatter(basic_formatter)


def set_debug_mode() -> None:
    """ set debug mode on """
    if config.DEBUG_MODE:
        logger.info(f'Start debug mode, log is stored in {config.LOGFILE_PATH}')
        init_fhlr(运行信息滚动窗)


def save_screenshot(img: bytes, subdir: str = '') -> None:
    """ save screenshot """
    if config.SCREENSHOT_PATH is None:
        return
    folder = Path(config.SCREENSHOT_PATH).joinpath(subdir)
    folder.mkdir(exist_ok=True, parents=True)
    if subdir != '-1' and len(list(folder.iterdir())) > config.SCREENSHOT_MAXNUM:
        screenshots = list(folder.iterdir())
        screenshots = sorted(screenshots, key=lambda x: x.name)
        for x in screenshots[: -config.SCREENSHOT_MAXNUM]:
            logger.debug(f'remove screenshot: {x.name}')
            x.unlink()
    filename = time.strftime('%Y%m%d%H%M%S.png', time.localtime())
    with folder.joinpath(filename).open('wb') as f:
        f.write(img)
    logger.debug(f'save screenshot: {filename}')


class log_sync(threading.Thread):
    """ recv output from subprocess """

    def __init__(self, process: str, pipe: int) -> None:
        self.process = process
        self.pipe = os.fdopen(pipe)
        super().__init__(daemon=True)

    def __del__(self) -> None:
        self.pipe.close()

    def run(self) -> None:
        while True:
            line = self.pipe.readline().strip()
            logger.debug(f'{self.process}: {line}')


def 日志设置():
    config.LOGFILE_PATH = 用户配置['日志存储目录']
    config.SCREENSHOT_PATH = 用户配置['截图存储目录']
    config.SCREENSHOT_MAXNUM = int(用户配置['每种截图的最大保存数量']) - 1
    config.MAX_RETRYTIME = 10
    日志全局格式 = '%(blue)s%(asctime)s %(white)s%(relativepath)s:%(lineno)d %(log_color)s%(funcName)s - %(message)s'
    for 日志格式 in logger.handlers:
        日志格式.setFormatter(colorlog.ColoredFormatter(日志全局格式, '%m-%d %H:%M:%S'))


class 项目经理(BaseSolver):
    服务器 = ''

    def __init__(self, device: 设备控制 = None, recog: Recognizer = None) -> None:
        super().__init__(device, recog)
        self.plan = None
        self.planned = None
        self.todo_task = None
        self.邮件设置 = None
        self.干员信息 = None
        self.跑单提前运行时间 = 300
        self.digit_reader = DigitReader()
        self.error = False
        self.任务列表 = []
        self.run_order_rooms = {}

    def 返回基建主界面(self):
        logger.info('返回基建主界面')
        返回次数 = 0
        while 返回次数 < 10 and not self.get_infra_scene() == 201:
            self.recog.update()
            if Mower0线程.stopped(): return
            if self.find('nav_button') is not None: self.tap((self.recog.w // 15, self.recog.h // 20))
            elif self.get_infra_scene() == 9998: time.sleep(3)
            else: self.back_to_infrastructure()
            self.recog.update()
            返回次数 += 1

    def run(self) -> None:
        if Mower0线程.stopped(): return
        self.error = False
        if len(self.任务列表) == 0:
            self.recog.update()
            time.sleep(1)
        self.处理报错(True)
        if len(self.任务列表) > 0: self.任务 = self.任务列表[0]
        else: self.任务 = None
        self.todo_task = False
        self.collect_notification = False
        self.planned = False
        if self.干员信息 is None or self.干员信息.operators is None: self.干员信息初始化()
        return super().run()

    def transition(self) -> None:
        self.recog.update()
        if self.get_infra_scene() == 1: self.tap_themed_element('index_infrastructure')
        elif self.scene() == 201: return self.基建主程序()
        elif self.scene() == 202: return self.收获()
        elif self.scene() == 205: self.back()
        elif self.scene() == 9998: time.sleep(1)
        elif self.scene() == 9: time.sleep(1)
        elif self.get_navigation(): self.tap_element('nav_infrastructure')
        elif self.scene() == 207: self.tap_element('arrange_blue_yes')
        elif self.get_infra_scene() == -1 or not self.scene() == -1:
            self.back_to_index()
            self.上个房间 = ''
        else: raise RecognizeError('Unknown scene')

    def find_next_task(self, compare_time=None, task_type='', compare_type='<'):
        if compare_type == '=':
            return next((e for e in self.任务列表 if the_same_time(e.time, compare_time) and (
                True if task_type == '' else task_type in e.type)), None)
        elif compare_type == '>':
            return next((e for e in self.任务列表 if (True if compare_time is None else e.time > compare_time) and (
                True if task_type == '' else task_type in e.type)), None)
        else:
            return next((e for e in self.任务列表 if (True if compare_time is None else e.time < compare_time) and (
                True if task_type == '' else task_type in e.type)), None)

    def 处理报错(self, force=False):
        global 循环次数, Mower0线程
        if Mower0线程.stopped(): return
        报错计时 = datetime.now()
        循环次数 = 0
        while self.scene() == -1:
            self.back_to_index()
            if 循环次数 < 10:
                time.sleep(1)
                self.recog.update()
                循环次数 += 1
                continue
            logger.error('处理报错')
            if (datetime.now() - 报错计时).total_seconds() > self.跑单提前运行时间 // 4:
                logger.error(f'报错次数达{循环次数}次，时间长达{round((datetime.now() - 报错计时).total_seconds())}秒')
                while not Mower0线程.stopped():
                    try:
                        Mower0线程._stop_event.set()
                        终止线程报错(Mower0线程.ident, SystemExit)
                    except: pass
                logger.warning('Mower0已停止，准备重新启动Mower0')
                Mower0线程 = 线程()
                Mower0线程.start()
                return
            self.recog.update()
        if self.error or force:
            # 如果没有任何时间小于当前时间的任务才生成空任务
            if self.find_next_task(datetime.now()) is None:
                logger.debug("由于出现错误情况，生成一次空任务来执行纠错")
                self.任务列表.append(SchedulerTask())
            # 如果没有任何时间小于当前时间的任务-10分钟 则清空任务
            if self.find_next_task(datetime.now() - timedelta(seconds=900)) is not None:
                logger.info("检测到执行超过15分钟的任务，清空全部任务")
                self.任务列表 = []
        elif self.find_next_task(datetime.now() + timedelta(hours=2.5)) is None:
            logger.debug("2.5小时内没有其他任务，生成一个空任务")
            self.任务列表.append(SchedulerTask(time=datetime.now() + timedelta(hours=2.5)))
        return True

    def 基建主程序(self):
        if Mower0线程.stopped(): return
        """ 位于基建首页 """
        if self.find('control_central') is None:
            self.back()
            return
        if self.任务 is not None:
            try:
                if len(self.任务.plan.keys()) > 0:
                    get_time = False
                    if "Shift_Change" == self.任务.type: get_time = True
                    self.跑单(self.任务.plan, get_time)
                    if get_time: self.plan_metadata()
                del self.任务列表[0]
            except Exception as e:
                logger.exception(e)
                self.跳过()
                self.error = True
            self.任务 = None
        elif not self.planned:
            try: self.任务调度器()
            except Exception as e:
                # 重新扫描
                self.error = True
                logger.exception({e})
            self.planned = True
        elif not self.todo_task:
            self.todo_task = True
        elif not self.collect_notification:
            公告 = detector.infra_notification(self.recog.img)
            if 公告 is None:
                time.sleep(1)
                公告 = detector.infra_notification(self.recog.img)
            if 公告 is not None: self.tap(公告)
            self.collect_notification = True
        else: return self.处理报错()

    def 任务调度器(self):
        global 工位表
        if Mower0线程.stopped(): return
        plan = self.plan
        # 如果下个 普通任务 <10 分钟则跳过 plan
        if self.find_next_task(datetime.now() + timedelta(seconds=600)) is not None: return
        if len(self.run_order_rooms) > 0:
            for 房间, 冗余信息 in self.run_order_rooms.items():
                if self.find_next_task(task_type=房间) is not None: continue;
                下次跑单任务 = {房间: ['Current'] * len(plan[房间])}
                for 序号, 位置 in enumerate(plan[房间]):
                    if '但书' in 位置['replacement'] or '龙舌兰' in 位置['replacement']:
                        下次跑单任务[房间][序号] = 位置['replacement'][0]
                # 点击进入该房间
                self.进入房间(房间)
                self.任务列表.append(SchedulerTask(time=self.读取接单时间(房间), task_plan=下次跑单任务, task_type=房间))
                if 工位表[房间][0] == '':
                    while self.find('arrange_check_in') is None:
                        self.recog.update()
                        if self.get_infra_scene() == 9:
                            if not self.waiting_solver(9, sleep_time=2, 服务器=服务器): self.返回基建主界面()
                        self.tap((self.recog.w // 15, self.recog.h // 20))
                    while self.find('room_detail') is None:
                        self.recog.update()
                        if self.get_infra_scene() == 9:
                            if not self.waiting_solver(9, sleep_time=2, 服务器=服务器): self.返回基建主界面()
                        if self.find('control_central') is None:
                            self.tap((self.recog.w // 20, self.recog.h * 2 // 5), interval=3)
                        else: self.进入房间(房间)
                    length = len(工位表[房间])
                    名字位置 = [((self.recog.w * 1460 // 1920, self.recog.h * 155 // 1080),
                                 (self.recog.w * 1700 // 1920, self.recog.h * 210 // 1080)),
                                ((self.recog.w * 1460 // 1920, self.recog.h * 370 // 1080),
                                 (self.recog.w * 1700 // 1920, self.recog.h * 420 // 1080)),
                                ((self.recog.w * 1460 // 1920, self.recog.h * 585 // 1080),
                                 (self.recog.w * 1700 // 1920, self.recog.h * 630 // 1080))]
                    该贸易站干员名单 = ''
                    for i in range(0, length):
                        读取到的干员名 = self.读取屏幕(self.recog.img[名字位置[i][0][1]:名字位置[i][1][1],
                                                       名字位置[i][0][0]:名字位置[i][1][0]], 模式="名字")
                        # 如果房间不为空
                        if not 读取到的干员名 == '':
                            if 读取到的干员名 not in self.干员信息.operators.keys() and 读取到的干员名 in agent_list:
                                工位表[房间][i] = 读取到的干员名
                                if not 该贸易站干员名单 == '': 该贸易站干员名单 += '、'
                                该贸易站干员名单 += 读取到的干员名
                    logger.info(f'记录干员工位：{该贸易站干员名单}')
        # 准备数据
        logger.debug(self.干员信息.print())

    def 干员信息初始化(self):
        plan = self.plan
        self.干员信息 = Operators({}, 4, plan)
        for room in plan.keys():
            for idx, data in enumerate(plan[room]):
                self.干员信息.add(Operator(data["agent"], room, idx, data["group"], data["replacement"], 'high',
                                           operator_type="high"))
        # 跑单
        for 门牌号, 排班表 in self.plan.items():
            if not 门牌号.startswith('room'): continue
            if any(('但书' in 位置['replacement'] or '龙舌兰' in 位置['replacement']) for 位置 in 排班表):
                self.run_order_rooms[门牌号] = {}

    def 读取接单时间(self, 门牌号):
        if Mower0线程.stopped(): return
        logger.info(f'读取贸易站 B{门牌号[5]}0{门牌号[7]} 接单时间')
        # 进入房间详情
        报错计数 = 0
        while self.find('bill_accelerate') is None:
            if 报错计数 > 5: raise Exception('未成功进入订单界面')
            self.tap((self.recog.w // 20, self.recog.h * 19 // 20), interval=3)
            报错计数 += 1
        接单等待时间 = self.统合读取时间((self.recog.w * 650 // 2496, self.recog.h * 660 // 1404,
                                          self.recog.w * 815 // 2496, self.recog.h * 710 // 1404), 读数器=True)
        logger.info(f'贸易站 B{门牌号[5]}0{门牌号[7]} 接单时间为 {接单等待时间.strftime("%H:%M:%S")}')
        跑单等待时间 = 接单等待时间 - timedelta(seconds=(self.跑单提前运行时间))
        return 跑单等待时间

    def 统合读取时间(self, 位置范围, 上限=None, 读数器=False):
        if 上限 is not None and 上限 < 36000: 上限 = 36000
        self.recog.update()
        总秒数 = self.读取时间(位置范围, 上限, 读数器)
        if 总秒数 is None: return datetime.now()
        行动时间 = datetime.now() + timedelta(seconds=(总秒数))
        return 行动时间

    def 飞桨初始化(self):
        det_model_dir = os.path.join(os.getcwd(), "paddleocr_models", "det")
        rec_model_dir = os.path.join(os.getcwd(), "paddleocr_models", "rec")
        cls_model_dir = os.path.join(os.getcwd(), "paddleocr_models", "cls")
        global ocr
        if ocr is None:
            # mac 平台不支持 mkldnn 加速，关闭以修复 mac 运行时错误
            if sys.platform == 'darwin':
                ocr = PaddleOCR(enable_mkldnn=False, use_angle_cls=False, cls=False, show_log=False, det_model_dir=det_model_dir, rec_model_dir=rec_model_dir, cls_model_dir=cls_model_dir)
            else: ocr = PaddleOCR(enable_mkldnn=True, use_angle_cls=False, cls=False, show_log=False, det_model_dir=det_model_dir, rec_model_dir=rec_model_dir, cls_model_dir=cls_model_dir)
            # ocr = PaddleOCR(enable_mkldnn=False, use_angle_cls=False, cls=False, show_log=False)

    def 读取屏幕(self, 图像, 模式="心情", 上限=24, 位置范围=None):
        if 位置范围 is not None:
            图像 = 图像[位置范围[1]:位置范围[3], 位置范围[0]:位置范围[2]]
        if '心情' in 模式 or 模式 == "时间":
            # 心情图片太小，复制8次提高准确率
            for 序号 in range(0, 4): 图像 = cv2.vconcat([图像, 图像])
        try:
            self.飞桨初始化()
            rets = ocr.ocr(图像, cls=False)
            line_conf = []
            for 序号 in range(len(rets[0])):
                res = rets[0][序号]
                if '心情' in 模式:
                    # 筛选掉不符合规范的结果
                    if ('/' + str(上限)) in res[1][0]: line_conf.append(res[1])
                else: line_conf.append(res[1])
            logger.debug(line_conf)
            if len(line_conf) == 0:
                if '心情' in 模式: return -1
                elif '名字' in 模式:
                    logger.debug("使用老版识别")
                    return character_recognize.agent_name(图像, self.recog.h)
                else: return ""
            x = [i[0] for i in line_conf]
            识别到的字符串 = max(set(x), key=x.count)
            if "心情" in 模式:
                if '.' in 识别到的字符串: 识别到的字符串 = 识别到的字符串.replace(".", "")
                心情值 = int(识别到的字符串[0:识别到的字符串.index('/')])
                return 心情值
            elif '时间' in 模式:
                if '.' in 识别到的字符串: 识别到的字符串 = 识别到的字符串.replace(".", ":")
            elif '名字' in 模式 and 识别到的字符串 not in agent_list:
                logger.debug("使用老版识别")
                识别到的字符串 = character_recognize.agent_name(图像, self.recog.h)
            logger.debug(识别到的字符串)
            return 识别到的字符串
        except Exception as e:
            logger.exception(e)
            return 上限 + 1

    def 读取时间(self, 位置范围, 上限, 报错计数=0, 读数器=False):
        # 刷新图片
        self.recog.update()
        if 读数器: 时间字符串 = self.digit_reader.get_time(self.recog.gray, self.recog.h, self.recog.w)
        else: 时间字符串 = self.读取屏幕(self.recog.img, 模式='时间', 位置范围=位置范围)
        try:
            时, 分, 秒 = str(时间字符串).split(':')
            if int(分) > 60 or int(秒) > 60: raise Exception(f"读取错误")
            折算秒数 = int(时) * 3600 + int(分) * 60 + int(秒)
            if 上限 is not None and 折算秒数 > 上限: raise Exception(f"超过读取上限")
            else: return 折算秒数
        except:
            logger.error("读取失败")
            time.sleep(3)
            self.tap((self.recog.w // 4, self.recog.h // 4), interval=1)
            if 报错计数 > 10:
                logger.exception(f"读取失败{报错计数}次超过上限")
                return None
            else: return self.读取时间(位置范围, 上限, 报错计数 + 1, 读数器)

    def 收获(self) -> None:
        """ 处理基建收获产物列表 """
        tapped = False
        干员信赖值 = self.find('infra_collect_trust')
        if 干员信赖值 is not None:
            logger.info('干员信赖')
            self.tap(干员信赖值)
            tapped = True
        订单 = self.find('infra_collect_bill')
        if 订单 is not None:
            logger.info('订单交付')
            self.tap(订单)
            tapped = True
        制造产物 = self.find('infra_collect_factory')
        if 制造产物 is not None:
            logger.info('可收获')
            self.tap(制造产物)
            tapped = True
        if not tapped:
            self.tap((self.recog.w // 20, self.recog.h * 19 // 20))
            self.todo_task = True

    def 进入房间(self, 门牌号: str) -> tp.Rectangle:
        if Mower0线程.stopped(): return
        """ 获取房间的位置并进入 """
        已进入房间 = False
        尝试进入次数 = 0
        while not 已进入房间:
            try:
                # 获取基建各个房间的位置
                基建房间划分 = segment.base(self.recog.img, self.find('control_central', strict=True))
                # 将画面外的部分删去
                _房间 = 基建房间划分[门牌号]

                for i in range(4):
                    _房间[i, 0] = max(_房间[i, 0], 0)
                    _房间[i, 0] = min(_房间[i, 0], self.recog.w)
                    _房间[i, 1] = max(_房间[i, 1], 0)
                    _房间[i, 1] = min(_房间[i, 1], self.recog.h)

                # 点击进入
                while self.find('control_central') is not None: self.tap(_房间[0], interval=3)
                if 门牌号.startswith('room'): logger.info(f'进入房间 B{门牌号[5]}0{门牌号[7]}')
                elif 门牌号 == 'dormitory_4': logger.info('进入房间 B401')
                else: logger.info(f'进入房间 B{门牌号[10]}04')
                已进入房间 = True
            except Exception as e:
                尝试进入次数 += 1
                self.返回基建主界面()
                time.sleep(3)
                if 尝试进入次数 > 5: raise e

    def 无人机协助跑单(self):
        logger.info('无人机协助跑单')
        while self.find('bill_accelerate') is not None: self.tap_element('bill_accelerate')
        点击计数 = 0
        while self.find('all_in') is not None and 点击计数 < 10:
            self.tap_element('all_in')
            点击计数 += 1
        while self.find('bill_accelerate') is None: self.tap((self.recog.w * 3 // 4, self.recog.h * 4 // 5))

    def 无人机加速调整订单时间(self, 门牌号: str, not_customize=False, not_return=False):
        # 点击进入该房间
        self.进入房间(门牌号)
        if self.get_infra_scene() == 9:
            if not self.waiting_solver(9, sleep_time=2, 服务器=服务器): return
        # 进入房间详情
        self.tap((self.recog.w // 20, self.recog.h * 19 // 20), interval=3)
        报错计数 = 0
        while self.find('factory_accelerate') is None and self.find('bill_accelerate') is None:
            if 报错计数 > 5: raise Exception('未成功进入无人机界面')
            self.tap((self.recog.w // 20, self.recog.h * 19 // 20), interval=3)
            报错计数 += 1
        无人机协助 = self.find('bill_accelerate')
        if 无人机协助:
            while (((self.任务列表[1].time - self.任务列表[0].time).total_seconds() < self.跑单提前运行时间
                   or 0 < (datetime.now().replace(hour=4, minute=0, second=0, microsecond=0) - 当前项目.任务列表[0].time).total_seconds() < 2 * 当前项目.跑单提前运行时间)
                   or 0 < (datetime.now().replace(hour=16, minute=0, second=0, microsecond=0) - 当前项目.任务列表[0].time).total_seconds() < 2 * 当前项目.跑单提前运行时间):
                logger.info(f'房间 B{门牌号[5]}0{门牌号[7]}')
                self.tap(无人机协助)
                if self.get_infra_scene() == 9:
                    if not self.waiting_solver(9, sleep_time=2, 服务器=服务器): return
                self.device.tap((self.recog.w * 1320 // 1920, self.recog.h * 502 // 1080))
                time.sleep(1)
                if self.get_infra_scene() == 9:
                    if not self.waiting_solver(9, sleep_time=2, 服务器=服务器): return
                self.tap((self.recog.w * 3 // 4, self.recog.h * 4 // 5))
                if self.get_infra_scene() == 9:
                    if not self.waiting_solver(9, sleep_time=2, 服务器=服务器): return
                while self.find('bill_accelerate') is None:
                    if 报错计数 > 5: raise Exception('未成功进入订单界面')
                    self.tap((self.recog.w // 20, self.recog.h * 19 // 20), interval=3)
                    报错计数 += 1
                加速后接单时间 = self.统合读取时间((self.recog.w * 650 // 2496, self.recog.h * 660 // 1404,
                                                    self.recog.w * 815 // 2496, self.recog.h * 710 // 1404),
                                                   读数器=True)
                self.任务列表[0].time = 加速后接单时间 - timedelta(seconds=(self.跑单提前运行时间))
                logger.info(
                    f'房间 B{门牌号[5]}0{门牌号[7]} 无人机加速后接单时间为 {加速后接单时间.strftime("%H:%M:%S")}')
                if not_customize:
                    无人机数量 = self.digit_reader.get_drone(self.recog.gray, self.recog.h, self.recog.w)
                    logger.info(f'当前无人机数量为 {无人机数量}')
                while self.find('bill_accelerate') is None:
                    if 报错计数 > 5: raise Exception('未成功进入订单界面')
                    self.tap((self.recog.w // 20, self.recog.h * 19 // 20), interval=3)
                    报错计数 += 1
        if not_return: return
        self.返回基建主界面()

    def get_arrange_order(self) -> 干员排序方式:
        best_score, best_order = 0, None
        for 顺序 in 干员排序方式:
            score = self.recog.score(干员排序方式位置[顺序][0])
            if score is not None and score[0] > best_score: best_score, best_order = score[0], 顺序
        logger.debug((best_score, best_order))
        return best_order

    def 切换干员排序方式(self, 方式序号: int, 倒序="false") -> None:
        self.tap((self.recog.w * 干员排序方式位置[干员排序方式(方式序号)][0],
                  self.recog.h * 干员排序方式位置[干员排序方式(方式序号)][1]), interval=0, rebuild=False)
        # 点个不需要的
        if 方式序号 < 4:
            self.tap((self.recog.w * 干员排序方式位置[干员排序方式(方式序号 + 1)][0],
                      self.recog.h * 干员排序方式位置[干员排序方式(方式序号)][1]), interval=0, rebuild=False)
        else:
            self.tap((self.recog.w * 干员排序方式位置[干员排序方式(方式序号 - 1)][0],
                      self.recog.h * 干员排序方式位置[干员排序方式(方式序号)][1]), interval=0, rebuild=False)
        # 切回来
        self.tap((self.recog.w * 干员排序方式位置[干员排序方式(方式序号)][0],
                  self.recog.h * 干员排序方式位置[干员排序方式(方式序号)][1]), interval=0.2, rebuild=True)
        if not 倒序 == "false":
            self.tap((self.recog.w * 干员排序方式位置[干员排序方式(方式序号)][0],
                      self.recog.h * 干员排序方式位置[干员排序方式(方式序号)][1]), interval=0.2, rebuild=True)

    def 查找干员(self, 查找干员列表: list[str], 报错计数=0, 最多干员总数=-1):
        try:
            # 识别干员
            self.recog.update()
            屏幕内识别到的干员 = character_recognize.agent(self.recog.img)  # 返回的顺序是从左往右从上往下
            # 提取识别出来的干员的名字
            选中干员列表 = []
            for 干员 in 屏幕内识别到的干员:
                干员名 = 干员[0]
                if 干员名 in 查找干员列表:
                    选中干员列表.append(干员名)
                    self.tap((干员[1][0]), interval=0)
                    查找干员列表.remove(干员名)
                    # 如果是按照个数选择 Free
                    if not 最多干员总数 == -1:
                        if len(选中干员列表) >= 最多干员总数: return 选中干员列表, 屏幕内识别到的干员
            return 选中干员列表, 屏幕内识别到的干员
        except Exception as e:
            报错计数 += 1
            if 报错计数 < 3:
                logger.exception(e)
                time.sleep(1)
                return self.查找干员(查找干员列表, 报错计数, 最多干员总数)
            else: raise e

    def 筛选器(self, 开, 模式="未进驻"):
        logger.info(f'开始 {("打开" if 开 else "关闭")} {模式} 筛选')
        self.tap((self.recog.w * 19 // 20, self.recog.h // 20), interval=1)
        if 模式 == "未进驻":
            未进驻 = self.find('arrange_non_check_in', score=0.9)
            if 开 ^ (未进驻 is None): self.tap((self.recog.w * 3 // 10, self.recog.h // 2), interval=0.5)
        # 确认
        self.tap((self.recog.w * 4 // 5, self.recog.h * 4 // 5), interval=0.5)

    def 换上干员(self, 换上干员列表: list[str], 门牌号: str) -> None:
        """
        :param order: 干员排序方式, 选择干员时右上角的排序功能
        """
        首位干员 = ''
        最大连续滑动次数 = 50
        for 序号, 干员名 in enumerate(换上干员列表):
            if 干员名 == '':
                self.换上干员(工位表[门牌号], 门牌号)
                self.返回基建主界面()
                重新运行Mower0()
                return
        当前换上干员列表 = copy.deepcopy(换上干员列表)
        换上干员名单 = str()
        for 干员名 in 当前换上干员列表:
            if not 换上干员名单 == '': 换上干员名单 += '、'
            换上干员名单 += 干员名
        if 门牌号.startswith('room') and ('但书' in 当前换上干员列表 or '龙舌兰' in 当前换上干员列表):
            logger.info(f'{换上干员名单} 进驻房间 B{门牌号[5]}0{门牌号[7]} 时间为 {(self.任务列表[0].time + timedelta(seconds=(self.跑单提前运行时间 - self.更换干员前缓冲时间))).strftime("%H:%M:%S")}')
        else: logger.info(f'换上 {换上干员名单}')
        刚进入干员选择界面 = True
        右移次数 = 0
        重试计数 = 0
        # 如果重复进入宿舍则需要排序
        选中干员列表 = []
        if 门牌号.startswith('room'): self.切换干员排序方式(2, "asc")
        else: self.切换干员排序方式(3, "asc")
        while len(当前换上干员列表) > 0:
            if Mower0线程.stopped(): return
            if 重试计数 > 3: raise Exception(f"到达最大尝试次数 3次")
            if 右移次数 > 最大连续滑动次数:
                # 到底了则返回再来一次
                for _ in range(右移次数):
                    self.swipe_only((self.recog.w // 2, self.recog.h // 2), (self.recog.w // 2, 0), interval=0.5)
                右移次数 = 0
                最大连续滑动次数 = 50
                重试计数 += 1
                self.筛选器(False)
            if 刚进入干员选择界面:
                self.tap((self.recog.w * 19 // 50, self.recog.h * 19 // 20), interval=0.5)
                当前选中干员列表, 屏幕中识别到的干员 = self.查找干员(当前换上干员列表)
                if 当前选中干员列表:
                    选中干员列表.extend(当前选中干员列表)
                    if len(当前换上干员列表) == 0: break
            刚进入干员选择界面 = False

            当前选中干员列表, 屏幕中识别到的干员 = self.查找干员(当前换上干员列表)
            if 当前选中干员列表: 选中干员列表.extend(当前选中干员列表)
            else:
                # 如果没找到 而且右移次数大于5
                if 屏幕中识别到的干员[0][0] == 首位干员 and 右移次数 > 5: 最大连续滑动次数 = 右移次数
                else: 首位干员 = 屏幕中识别到的干员[0][0]
                st = 屏幕中识别到的干员[-2][1][2]  # 起点
                ed = 屏幕中识别到的干员[0][1][1]  # 终点
                self.swipe_noinertia(st, (ed[0] - st[0], 0))
                右移次数 += 1
            if len(当前换上干员列表) == 0: break;

        # 排序
        if not len(换上干员列表) == 1:
            # 左移
            self.左移(右移次数, self.recog.w, self.recog.h)
            self.tap((self.recog.w * 干员排序方式位置[干员排序方式.技能][0],
                      self.recog.h * 干员排序方式位置[干员排序方式.技能][1]), interval=0.5, rebuild=False)
            相对位置列表 = [(0.35, 0.35), (0.35, 0.75), (0.45, 0.35), (0.45, 0.75), (0.55, 0.35)]
            不匹配 = False
            for 序号, item in enumerate(换上干员列表):
                if not 换上干员列表[序号] == 选中干员列表[序号] or 不匹配:
                    不匹配 = True
                    p_idx = 选中干员列表.index(换上干员列表[序号])
                    self.tap((self.recog.w * 相对位置列表[p_idx][0], self.recog.h * 相对位置列表[p_idx][1]),
                             interval=0.5, rebuild=False)
                    self.tap((self.recog.w * 相对位置列表[p_idx][0], self.recog.h * 相对位置列表[p_idx][1]),
                             interval=0.5, rebuild=False)
        self.上个房间 = 门牌号

    def 左移(self, 滑动次数, 水平距离, 竖直距离):
        for 滑动 in range(滑动次数): self.swipe_only((水平距离 // 2, 竖直距离 // 2), (水平距离 // 2, 0), interval=0.5)
        return 0

    @push_operators
    def 撤下干员(self, 门牌号, 读取时间参数=None):
        if Mower0线程.stopped(): return
        if 读取时间参数 is None: 读取时间参数 = []
        场合报错 = 0
        while self.find('room_detail') is None:
            if self.get_infra_scene() == 9:
                if not self.waiting_solver(9, sleep_time=2, 服务器=服务器): return
            if self.find('control_central') is None:
                self.tap((self.recog.w // 20, self.recog.h * 2 // 5), interval=0.5)
            else: self.进入房间(门牌号)
            if 场合报错 > 5: raise Exception('未成功进入进驻信息界面')
            场合报错 += 1
        length = len(self.plan[门牌号])
        if length > 3:
            self.swipe((self.recog.w * 4 // 5, self.recog.h // 2), (0, self.recog.h * 9 // 20), duration=500,
                       interval=1, rebuild=True)
        名字位置 = [((self.recog.w * 1460 // 1920, self.recog.h * 155 // 1080),
                     (self.recog.w * 1700 // 1920, self.recog.h * 210 // 1080)),
                    ((self.recog.w * 1460 // 1920, self.recog.h * 370 // 1080),
                     (self.recog.w * 1700 // 1920, self.recog.h * 420 // 1080)),
                    ((self.recog.w * 1460 // 1920, self.recog.h * 585 // 1080),
                     (self.recog.w * 1700 // 1920, self.recog.h * 630 // 1080)),
                    ((self.recog.w * 1460 // 1920, self.recog.h * 560 // 1080),
                     (self.recog.w * 1700 // 1920, self.recog.h * 610 // 1080)),
                    ((self.recog.w * 1460 // 1920, self.recog.h * 775 // 1080),
                     (self.recog.w * 1700 // 1920, self.recog.h * 820 // 1080))]
        结果 = []
        滑动后 = False
        for i in range(0, length):
            if i >= 3 and not 滑动后:
                self.swipe((self.recog.w * 4 // 5, self.recog.h // 2), (0, -self.recog.h * 9 // 20), duration=500,
                           interval=1, rebuild=True)
                滑动后 = True
            数据 = {}
            读取到的干员名 = self.读取屏幕(self.recog.img[名字位置[i][0][1]:名字位置[i][1][1], 名字位置[i][0][0]:名字位置[i][1][0]], 模式="名字")
            场合报错 = 0
            while i >= 3 and not 读取到的干员名 == '' and (next((e for e in 结果 if e['agent'] == 读取到的干员名), None)) is not None:
                logger.warning("检测到滑动可能失败")
                self.swipe((self.recog.w * 4 // 5, self.recog.h // 2),(0, -self.recog.h * 9 // 20), duration=500, interval=1, rebuild=True)
                读取到的干员名 = self.读取屏幕(
                    self.recog.img[名字位置[i][0][1]:名字位置[i][1][1], 名字位置[i][0][0]:名字位置[i][1][0]], 模式="名字")
                场合报错 += 1
                if 场合报错 > 1: raise Exception("超过出错上限")
            # 如果房间不为空
            if not 读取到的干员名 == '':
                if 读取到的干员名 not in self.干员信息.operators.keys() and 读取到的干员名 in agent_list:
                    self.干员信息.add(Operator(读取到的干员名, ""))
                更新时间 = False
                if self.干员信息.operators[读取到的干员名].need_to_refresh(r=门牌号): 更新时间 = True
                high_no_time = self.干员信息.update_detail(读取到的干员名, 24, 门牌号, i, 更新时间)
                数据['depletion_rate'] = self.干员信息.operators[读取到的干员名].depletion_rate
            数据['agent'] = 读取到的干员名
            if i in 读取时间参数:
                数据['time'] = datetime.now()
                self.干员信息.refresh_dorm_time(门牌号, i, 数据)
                logger.debug(f"停止记录时间:{str(数据)}")
            结果.append(数据)
        撤下干员名单 = '撤下'
        for _operator in self.干员信息.operators.keys():
            if self.干员信息.operators[_operator].current_room == 门牌号 and _operator not in [res['agent'] for res in 结果]:
                self.干员信息.operators[_operator].current_room = ''
                self.干员信息.operators[_operator].current_index = -1
                if 撤下干员名单 == '撤下': 撤下干员名单 += ' '
                else: 撤下干员名单 += '、'
                撤下干员名单 += _operator
        if not 撤下干员名单 == '撤下': logger.info(撤下干员名单)
        return 结果

    def 刷新当前房间干员列表(self, 门牌号):
        _当前房间干员列表 = self.干员信息.get_current_room(门牌号)
        if _当前房间干员列表 is None:
            self.撤下干员(门牌号)
            _当前房间干员列表 = self.干员信息.get_current_room(门牌号, True)
        return _当前房间干员列表

    def 检查换人情况(self, 房间, 任务列表, 获取时间=False):
        读取时间参数 = []
        if 获取时间: 读取时间参数 = self.干员信息.get_refresh_index(房间, 任务列表[房间])
        当前房间干员列表 = self.撤下干员(房间, 读取时间参数)
        for 序号, 干员名 in enumerate(任务列表[房间]):
            if not 当前房间干员列表[序号]['agent'] == 干员名:
                logger.error(f'检测到的干员{当前房间干员列表[序号]["agent"]},需要上岗的干员{干员名}')
                raise Exception('检测到安排干员未成功')

    def 跑单(self, 任务列表: tp.BasePlan, 获取时间=False):
        # global 上次跑单时间
        任务房间列表 = list(任务列表.keys())
        换回上班干员 = False
        # 优先替换工作站再替换宿舍
        # 任务房间列表.sort(key=lambda x: x.startswith('dorm'), reverse=False)
        for 房间 in 任务房间列表:
            干员已进驻 = False
            干员选择报错 = 0
            while not 干员已进驻:
                try:
                    场合报错 = 0
                    self.进入房间(房间)
                    while self.find('room_detail') is None:
                        self.recog.update()
                        if self.get_infra_scene() == 9:
                            if not self.waiting_solver(9, sleep_time=2, 服务器=服务器): return
                        if self.find('control_central') is None:
                            self.tap((self.recog.w // 20, self.recog.h * 2 // 5), interval=3)
                        else: self.进入房间(房间)
                        if 场合报错 > 5: raise Exception('未成功进入进驻信息界面')
                        场合报错 += 1
                    if 干员选择报错 == 0:
                        if 'Current' in 任务列表[房间]:
                            self.刷新当前房间干员列表(房间)
                            for 序号, _当前干员名 in enumerate(任务列表[房间]):
                                if _当前干员名 == 'Current':
                                    任务列表[房间][序号] = self.干员信息.get_current_room(房间, True)[序号]
                    while self.find('arrange_order_options') is None:
                        self.recog.update()
                        if 场合报错 > 10: raise Exception('未成功进入干员选择界面')
                        if self.find('room_detail') is not None:
                            self.tap((self.recog.w * 41 // 50, self.recog.h // 5), interval=3)
                        场合报错 += 1
                    换回上班干员 = True
                    self.换上干员(任务列表[房间], 房间)
                    self.recog.update()
                    if 房间.startswith('room') and 用户配置['跑单消耗无人机开关'] == '关':
                        龙舌兰_但书进驻前的等待时间 = round(((self.任务列表[0].time - datetime.now()).total_seconds()
                                                             + self.跑单提前运行时间 - self.更换干员前缓冲时间), 1)
                        if 龙舌兰_但书进驻前的等待时间 > 0:
                            logger.info(f'龙舌兰、但书进驻前等待 {str(龙舌兰_但书进驻前的等待时间)} 秒')
                            time.sleep(龙舌兰_但书进驻前的等待时间)
                    self.tap_element('confirm_blue', detected=True, judge=False, interval=3)
                    self.recog.update()
                    if self.get_infra_scene() == 206:
                        self.tap((self.recog.w * 2 // 3, self.recog.h - 10), rebuild=True)
                    self.检查换人情况(房间, 任务列表, 获取时间)
                    干员已进驻 = True
                except Exception as e:
                    logger.exception(e)
                    干员选择报错 += 1
                    场合报错 = 0
                    self.进入房间(房间)
                    while self.find('room_detail') is None:
                        self.recog.update()
                        if self.get_infra_scene() == 9:
                            if not self.waiting_solver(9, sleep_time=2, 服务器=服务器): return
                        if self.find('control_central') is None:
                            self.tap((self.recog.w // 20, self.recog.h * 2 // 5), interval=3)
                        else: self.进入房间(房间)
                        if 场合报错 > 5: raise Exception('未成功进入进驻信息界面')
                        场合报错 += 1
                    while self.find('arrange_order_options') is None:
                        self.recog.update()
                        if 场合报错 > 10: raise Exception('未成功进入干员选择界面')
                        if self.find('room_detail') is not None:
                            self.tap((self.recog.w * 41 // 50, self.recog.h // 5), interval=3)
                        场合报错 += 1
                    self.换上干员(工位表[房间], 房间)
                    self.返回基建主界面()
            # 上次跑单时间 = self.任务列表[0].time
            del 任务列表[房间]
            if 房间.startswith('room'):
                场合报错 = 0
                while self.find('bill_accelerate') is None:
                    if 场合报错 > 5: raise Exception('未成功进入订单界面')
                    self.tap((self.recog.w // 20, self.recog.h * 19 // 20), interval=1)
                    场合报错 += 1

                修正后的接单时间 = self.统合读取时间((self.recog.w * 650 // 2496, self.recog.h * 660 // 1404,
                                                      self.recog.w * 815 // 2496, self.recog.h * 710 // 1404), 读数器=True)
                等待时间 = round((修正后的接单时间 - datetime.now()).total_seconds(), 1)
                if (等待时间 > 0) and (等待时间 < self.跑单提前运行时间 * 2):
                    try:
                        if 用户配置['跑单消耗无人机开关'] == '关':
                            logger.info(f'房间 B{房间[5]}0{房间[7]} 修正后的接单时间为 {修正后的接单时间.strftime("%H:%M:%S")}')
                            logger.info(f'等待截图时间为 {str(等待时间)} 秒')
                            time.sleep(等待时间)
                        else: self.无人机协助跑单()
                        self.recog.update()
                        while self.find("order_ready", scope=((self.recog.w * 450 // 1920, self.recog.h * 675 // 1080), (self.recog.w * 600 // 1920, self.recog.h * 750 // 1080))) is None:
                            time.sleep(1)
                            logger.info('等待截图时间 +1秒')
                            self.recog.update()
                        logger.info('跑单成功')
                    except Exception as e: logger.exception(e)
                else: logger.debug('检测到漏单！')
                logger.info('保存截图')
                if self.get_infra_scene() == 9:
                    if not self.waiting_solver(9, sleep_time=2, 服务器=服务器): return
                self.recog.save_screencap('run_order')
            if self.get_infra_scene() == 9:
                if not self.waiting_solver(9, sleep_time=2, 服务器=服务器): return

            if 换回上班干员 and self.任务列表[0].type.startswith('room'):
                while 换回上班干员:
                    try:
                        场合报错 = 0
                        while self.find('arrange_order_options') is None:
                            if 场合报错 > 5: raise Exception('未成功进入干员选择界面')
                            if self.find('control_central') is None:
                                self.tap((self.recog.w * 41 // 50, self.recog.h // 5), interval=1)
                                self.tap((self.recog.w // 4, self.recog.h * 9 // 10), interval=1)
                            else: self.进入房间(房间)
                            场合报错 += 1
                        self.换上干员(工位表[房间], 房间)
                        self.recog.update()
                        self.tap_element('confirm_blue', detected=True, judge=False, interval=3)
                        self.recog.update()
                        if self.get_infra_scene() == 206: self.tap((self.recog.w * 2 // 3, self.recog.h - 10), rebuild=True)
                        logger.info("订单交付")
                        场合报错 = 0
                        while (self.find(
                                "order_ready", scope=((self.recog.w * 450 // 1920, self.recog.h * 675 // 1080),
                                                      (self.recog.w * 600 // 1920, self.recog.h * 750 // 1080)))
                               is not None):
                            if 场合报错 > 5: raise Exception('未成功交付订单')
                            self.tap((self.recog.w // 4, self.recog.h // 4), interval=1)
                            场合报错 += 1
                        self.back(interval=2)
                        场合报错 = 0
                        while self.find('room_detail') is None:
                            if self.get_infra_scene() == 9:
                                if not self.waiting_solver(9, sleep_time=2, 服务器=服务器): return
                            if self.find('control_central') is None:
                                self.tap((self.recog.w // 20, self.recog.h * 2 // 5), interval=0.5)
                            else: self.进入房间(房间)
                            if 场合报错 > 5: raise Exception('未成功进入进驻信息界面')
                            场合报错 += 1
                        self.检查换人情况(房间, 工位表, 获取时间)
                        换回上班干员 = False
                    except Exception as e:
                        logger.exception(e)
                        self.返回基建主界面()

                if 龙舌兰和但书休息 and 房间.startswith('room'):
                    宿舍 = {}
                    宿舍[龙舌兰和但书休息宿舍] = [data["agent"] for data in self.plan[龙舌兰和但书休息宿舍]]
                    self.任务列表.append(SchedulerTask(time=self.任务列表[0].time, task_plan=宿舍))


    def 跳过(self, 任务种类='All'):
        if 任务种类 == 'All': 任务种类 = ['planned', 'collect_notification', 'todo_task']
        if 'planned' in 任务种类: self.planned = True
        if 'todo_task': self.todo_task = True
        if 'collect_notification': self.collect_notification = True

    @CFUNCTYPE(None, c_int, c_char_p, c_void_p)
    def MAA日志(MAA日志, details, arg):
        日志 = Message(MAA日志)
        日志内容 = json.loads(details.decode('utf-8'))
        logger.debug(日志内容)
        logger.debug(日志)
        logger.debug(arg)
        if "what" in 日志内容 and 日志内容["what"] == "StageDrops":
            global 关卡掉落
            关卡掉落["details"].append(日志内容["details"]["drops"])
            关卡掉落["summary"] = 日志内容["details"]["stats"]
            for 种类 in 日志内容["details"]["drops"]:
                物品名称 = 种类["itemName"]
                物品数量 = 种类["quantity"]
                logger.info(f"关卡掉落: {物品名称} × {物品数量}")

    def MAA初始化(self):
        MAA路径 = pathlib.Path(self.MAA设置['MAA路径'])
        if MAA路径.suffix == '.exe':
            MAA路径 = MAA路径.parent
        asst_path = os.path.dirname(MAA路径 / "Python" / "asst")
        if asst_path not in sys.path: sys.path.append(asst_path)

        from asst.asst import Asst
        global Message
        from asst.utils import Message

        Asst.load(path=self.MAA设置['MAA路径'])
        self.MAA = Asst(callback=self.MAA日志)
        self.关卡列表 = []
        if self.MAA.connect(self.MAA设置['MAA_adb路径'], self.device.client.device_id): logger.info("MAA 连接成功")
        else:
            logger.info("MAA 连接失败")
            raise Exception("MAA 连接失败")

    def 添加MAA任务(self, type):
        if type in ['StartUp', 'Visit', 'Award']: self.MAA.append_task(type)
        elif type == 'Fight':
            关卡 = self.MAA设置['消耗理智关卡']
            if 关卡 == '上一次作战': 关卡 = ''
            self.MAA.append_task('Fight', {
                'stage': 关卡,
                'medicine': int(MAA设置['使用理智药数量']),
                'stone': 0,
                'times': 999,
                'report_to_penguin': True,
                'client_type': '',
                'penguin_id': str(用户配置.get('企鹅物流汇报ID', '')),
                'DrGrandet': False,
                'server': 'CN',
                'expiring_medicine': 9999
            })
            self.关卡列表.append(关卡)
        # elif type == 'Recruit':
        #     self.MAA.append_task('Recruit', {
        #         'select': [4],
        #         'confirm': [3, 4],
        #         'times': 4,
        #         'refresh': True,
        #         "recruitment_time": {
        #             "3": 460,
        #             "4": 540
        #         }
        #     })
        # elif type == 'Mall':
        #     credit_fight = False
        #     if len(self.关卡列表) > 0 and self.关卡列表[- 1] != '':
        #         credit_fight = True
        #     self.MAA.append_task('Mall', {
        #         'shopping': True,
        #         'buy_first': ['招聘许可'],
        #         'blacklist': ['家具', '碳', '加急许可'],
        #         'credit_fight': credit_fight
        #     })

    def MAA任务调度器(self, 任务列表=['Fight'], 首次=False):
        try:
            self.发送邮件('启动MAA')
            self.back_to_index()
            # 任务及参数请参考 docs/集成文档.md
            self.MAA初始化()
            if 任务列表 == 'All': 任务列表 = ['StartUp', 'Fight', 'Recruit', 'Visit', 'Mall', 'Award']

            if self.MAA设置['集成战略'] == '开' or self.MAA设置['生息演算'] == '开':
                while (self.任务列表[0].time - datetime.now()).total_seconds() > 30:
                    self.MAA = None
                    self.MAA初始化()
                    主题 = str()
                    if self.MAA设置['集成战略'] == '开':
                        if self.MAA设置['集成战略主题'] == '傀影与猩红孤钻': 主题 = 'Phantom'
                        elif self.MAA设置['集成战略主题'] == '水月与深蓝之树': 主题 = 'Mizuki'
                        elif self.MAA设置['集成战略主题'] == '探索者的银凇止境': 主题 = 'Sami'
                        self.MAA.append_task('Roguelike', {
                            'theme': 主题,
                            'mode': int(self.MAA设置['集成战略策略模式']),
                            'starts_count': 9999999,
                            'investment_enabled': True,
                            'investments_count': 9999999,
                            'stop_when_investment_full': False,
                            'squad': self.MAA设置['集成战略分队'],
                            'roles': self.MAA设置['集成战略开局招募组合'],
                            'core_char': self.MAA设置['集成战略开局干员']
                        })
                    elif self.MAA设置['生息演算'] == '开':
                        self.back_to_reclamation_algorithm()
                        self.MAA.append_task('ReclamationAlgorithm')
                    # elif self.MAA设置['保全派驻'] :
                    #     self.MAA.append_task('SSSCopilot', {
                    #         'filename': "F:\\MAA-v4.10.5-win-x64\\resource\\copilot\\SSS_阿卡胡拉丛林.json",
                    #         'formation': False,
                    #         'loop_times':99
                    #     })
                    self.MAA.start()
                    while self.MAA.running():
                        if (self.任务列表[0].time - datetime.now()).total_seconds() < 30:
                            self.MAA.stop()
                            break
                        else: time.sleep(0)
                    self.device.exit(self.服务器)
            else:
                for MAA任务 in 任务列表: self.添加MAA任务(MAA任务)
                # asst.append_task('Copilot', {
                #     'stage_name': '千层蛋糕',
                #     'filename': './GA-EX8-raid.json',
                #     'formation': False

                # })
                self.MAA.start()
                MAA停止时间 = None
                if 首次: MAA停止时间 = datetime.now() + timedelta(minutes=5)
                else:
                    global 关卡掉落
                    关卡掉落 = {"details": [], "summary": {}}
                logger.info(f"MAA 启动")
                强制停止MAA = False
                while self.MAA.running():
                    # 单次任务默认5分钟
                    if 首次 and MAA停止时间 < datetime.now():
                        self.MAA.stop()
                        强制停止MAA = True
                    # 5分钟之前就停止
                    elif not 首次 and (self.任务列表[0].time - datetime.now()).total_seconds() < 300:
                        self.MAA.stop()
                        强制停止MAA = True
                    else: time.sleep(0)
                self.发送邮件('MAA 停止')
                if 强制停止MAA:
                    logger.info(f"MAA 任务未完成，等待3分钟重启软件")
                    time.sleep(180)
                    self.device.exit(self.服务器)
                elif not 首次:
                    logger.info(f"记录 MAA 本次执行时间:{datetime.now()}")
                    # 有掉落东西再发
                    if 关卡掉落["details"]: self.发送邮件(maa_template.render(stage_drop=关卡掉落), "Maa停止")
                else: self.发送邮件("Maa单次任务停止")

            if 首次:
                if len(self.任务列表) > 0: del self.任务列表[0]
                self.MAA = None
                if self.find_next_task(datetime.now() + timedelta(seconds=900)) is None:
                    # 未来10分钟没有任务就新建
                    self.任务列表.append(SchedulerTask())
                return
            任务倒计时 = (下个任务开始时间 - datetime.now()).total_seconds()
            if 任务倒计时 > 0:
                logger.info(f"休息{int(任务倒计时)}秒，到{下个任务开始时间}")
                time.sleep(任务倒计时)
            self.MAA = None
        except Exception as e:
            logger.error(e)
            self.MAA = None
            self.device.exit(self.服务器)
            任务倒计时 = (下个任务开始时间 - datetime.now()).total_seconds()
            if 任务倒计时 > 0:
                logger.info(f"休息{int(任务倒计时 / 60)}分钟，到{下个任务开始时间}")
                time.sleep(任务倒计时)

    def 发送邮件(self, 内容=None, 主题='', 重试次数=3):
        global 任务
        if not self.邮件设置['邮件提醒开关'] == '开':
            logger.info('邮件功能未开启')
            return
        while 重试次数 > 0:
            try:
                邮件 = MIMEMultipart()
                if 内容 is None:
                    内容 = """
                    <html>
                        <body>
                        <table border="1">
                        <tr><th>时间</th><th>房间</th></tr>                    
                    """
                    for 任务 in self.任务列表:
                        内容 += f"""
                                        <tr><td>{任务.time.strftime('%Y-%m-%d %H:%M:%S')}</td>
                                        <td>B{任务.type[5]}0{任务.type[7]}</td></tr>    
                                    """
                    内容 += "</table></body></html>\n"
                    # 森空岛信息
                    if 森空岛小秘书:
                        森空岛信息 = _信息内容.split('\n')
                        内容 += """
                        <html>
                            <body>
                            <table border="1">                  
                        """
                        for 行 in 森空岛信息:
                            if len(行) > 1: 内容 += f"<tr><td>{行}</td></tr>"
                        内容 += "</table></body></html>"
                    邮件.attach(MIMEText(内容, 'html'))
                else: 邮件.attach(MIMEText(str(内容), 'plain', 'utf-8'))
                邮件['Subject'] = (f"将在{self.任务列表[0].time.strftime('%H:%M')}于房间 "
                                   f"B{self.任务列表[0].type[5]}0{self.任务列表[0].type[7]} 进行跑单")
                邮件['From'] = self.邮件设置['发信邮箱']
                邮箱 = smtplib.SMTP_SSL("smtp.qq.com", 465, timeout=10.0)
                # 登录邮箱
                邮箱.login(self.邮件设置['发信邮箱'], self.邮件设置['授权码'])
                # 开始发送
                邮箱.sendmail(self.邮件设置['发信邮箱'], self.邮件设置['收件人邮箱'], 邮件.as_string())
                break
            except Exception as e:
                logger.error("邮件发送失败")
                logger.exception(e)
                重试次数 -= 1
                time.sleep(1)


def 初始化(任务列表, scheduler=None):
    global 工位表
    device = 设备控制()
    cli = Solver(device)
    if scheduler is None:
        当前项目 = 项目经理(cli.device, cli.recog)
        logger.info(f'当前模拟器分辨率为 {当前项目.recog.w} × {当前项目.recog.h}')
        if not 当前项目.recog.w * 9 == 当前项目.recog.h * 16:
            logger.error('请将模拟器分辨率设置为 16:9 再重新运行 Mower0!')
            托盘图标.notify('请将模拟器分辨率设置为 16:9 \n再重新运行 Mower0!', "分辨率检验")
            停止运行Mower0()
        当前项目.服务器 = 服务器
        当前项目.operators = {}
        当前项目.plan = {}
        当前项目.current_base = {}
        for 房间 in 用户配置['跑单位置设置']:
            当前项目.plan[f'room_{房间[1]}_{房间[3]}'] = []
            if not f'room_{房间[1]}_{房间[3]}' in 工位表: 工位表[f'room_{房间[1]}_{房间[3]}'] = []
            for 干员 in 用户配置['跑单位置设置'][房间]:
                当前项目.plan[f'room_{房间[1]}_{房间[3]}'].append({'agent': '', 'group': '', 'replacement': [干员]})
                if len(工位表[f'room_{房间[1]}_{房间[3]}']) == 0 or 工位表[f'room_{房间[1]}_{房间[3]}'][0] == '':
                    工位表[f'room_{房间[1]}_{房间[3]}'].append('')
        if 龙舌兰和但书休息:
            global 龙舌兰和但书休息宿舍
            for 宿舍 in 用户配置['宿舍设置']:
                if 宿舍 == 'B401': 龙舌兰和但书休息宿舍 = 'dormitory_4'
                else: 龙舌兰和但书休息宿舍 = 'dormitory_' + 宿舍[1]
                当前项目.plan[龙舌兰和但书休息宿舍] = []
                for 干员 in 用户配置['宿舍设置'][宿舍]:
                    if 干员 == '当前休息干员': 干员 = 'Current'
                    if 干员 == '自动填充干员': 干员 = 'Free'
                    当前项目.plan[龙舌兰和但书休息宿舍].append({'agent': 干员, 'group': '', 'replacement': ''})
        当前项目.任务列表 = 任务列表
        当前项目.上个房间 = ''
        当前项目.MAA = None
        当前项目.邮件设置 = 邮件设置
        当前项目.ADB_CONNECT = config.ADB_CONNECT[0]
        当前项目.MAA设置 = MAA设置
        当前项目.error = False
        当前项目.跑单提前运行时间 = 跑单提前运行时间
        当前项目.更换干员前缓冲时间 = 更换干员前缓冲时间
        return 当前项目
    else:
        scheduler.device = cli.device
        scheduler.recog = cli.recog
        scheduler.处理报错(True)
        return scheduler


class 线程(threading.Thread):

    def __init__(self, dae=False, *args, **kwargs):
        super(线程, self).__init__(daemon=dae, *args, **kwargs)
        self._stop_event = threading.Event()

    def stopped(self):
        return self._stop_event.is_set()

    def run(self):
        self.Mower0()

    def Mower0(self):
        global ope_list, 当前项目, 任务提示, 下个任务开始时间, 已签到日期
        # 第一次执行任务
        任务列表 = []
        for Mower0任务 in 任务列表: Mower0任务.time = datetime.strptime(str(Mower0任务.time), '%m-%d %H:%M:%S.%f')
        重连次数上限 = 10
        重连次数 = 0
        当前项目 = 初始化(任务列表)
        当前项目.device.launch(f"{服务器}/{config.APP_ACTIVITY_NAME}")
        if 签到: 森空岛签到()
        当前项目.干员信息初始化()
        while True:
            if self.stopped(): break
            try:
                if len(当前项目.任务列表) > 0:
                    当前项目.任务列表.sort(key=lambda x: x.time, reverse=False)  # 任务按时间排序
                    # 如果订单间的时间差距小，无人机协助拉开订单间的时间差距
                    if (len(任务列表) > 1 and (任务列表[0].time - datetime.now()).total_seconds()
                            > 当前项目.跑单提前运行时间 > (任务列表[1].time - 任务列表[0].time).total_seconds()):
                        logger.warning("两个订单时间太接近了，准备用无人机协助拉开订单间时间差距")
                        当前项目.无人机加速调整订单时间(任务列表[0].type, True, True)
                    # 如果开始跑单时间到4:00或16:00的差距过小，无人机协助提前订单时间
                    if (len(任务列表) > 1 and 2 * 当前项目.跑单提前运行时间 >
                            (datetime.now().replace(hour=4, minute=0, second=0, microsecond=0)
                             - 当前项目.任务列表[0].time).total_seconds() > 0):
                        logger.warning("跑单时间与4:00太接近了，准备用无人机协助提前订单时间")
                        当前项目.无人机加速调整订单时间(任务列表[0].type, True, True)
                    if (len(任务列表) > 1 and 2 * 当前项目.跑单提前运行时间 >
                            (datetime.now().replace(hour=16, minute=0, second=0, microsecond=0)
                             - 当前项目.任务列表[0].time).total_seconds() > 0):
                        logger.warning("跑单时间与16:00太接近了，准备用无人机协助提前订单时间")
                        当前项目.无人机加速调整订单时间(任务列表[0].type, True, True)

                    下个任务开始时间 = 任务列表[0].time
                    当前项目.返回基建主界面()
                    任务间隔 = (当前项目.任务列表[0].time - datetime.now()).total_seconds()
                    if 任务间隔 > 0:
                        try:
                            if 森空岛小秘书: 森空岛查看游戏内信息()
                            if 签到 and not 已签到日期 == datetime.now().strftime('%Y年%m月%d日'): 森空岛签到()
                        except: pass
                        当前项目.发送邮件()
                        任务提示 = str()
                        for 任务序号 in range(len(任务列表)):
                            logger.warning(f'房间 B{任务列表[任务序号].type[5]}0{任务列表[任务序号].type[7]} '
                                           f'开始跑单的时间为 {任务列表[任务序号].time.strftime("%H:%M:%S")}')
                        无人机数量 = 当前项目.digit_reader.get_drone(当前项目.recog.gray, 当前项目.recog.h, 当前项目.recog.w)
                        if 无人机数量 > 160:
                            logger.warning(f'现在有 {无人机数量} 个无人机，请尽快使用，避免溢出！')
                            任务提示 += f'现在有 {无人机数量} 个无人机，请尽快使用！\n'
                        for 任务序号 in range(len(任务列表)):
                            任务提示 += (f'房间 B{任务列表[任务序号].type[5]}0{任务列表[任务序号].type[7]} '
                                         f'开始跑单的时间为 {任务列表[任务序号].time.strftime("%H:%M:%S")}\n')

                        # 如果有高强度重复MAA任务,任务间隔超过10分钟则启动MAA
                        if MAA设置['作战开关'] == '开' and (任务间隔 > 600): 当前项目.MAA任务调度器()
                        else:
                            if 用户配置['任务结束后退出游戏'] == '是' and 任务间隔 > 跑单提前运行时间:
                                当前项目.device.exit(当前项目.服务器)
                            else: 当前项目.back_to_index()
                            if 森空岛小秘书:
                                while (当前项目.任务列表[0].time - datetime.now()).total_seconds() > 2 * 跑单提前运行时间 + 360:
                                    if self.stopped(): return
                                    try:
                                        time.sleep(360)
                                        森空岛实时数据分析()
                                    except: pass
                            else: time.sleep(max(任务间隔 - 跑单提前运行时间, 0))
                            if self.stopped(): return
                            time.sleep(max((当前项目.任务列表[0].time - datetime.now()).total_seconds() - 跑单提前运行时间, 0))
                            if 弹窗提醒: 托盘图标.notify("跑单时间快到了喔，请放下游戏中正在做的事，或者手动关闭Mower0", "Mower0跑单提醒")
                            time.sleep(max((当前项目.任务列表[0].time - datetime.now()).total_seconds(), 0))
                            logger.info(f'房间 B{任务列表[0].type[5]}0{任务列表[0].type[7]} 开始跑单')
                            if 弹窗提醒: 托盘图标.notify("开始跑单！", "Mower0跑单提醒")
                    当前项目.device.launch(f"{服务器}/{config.APP_ACTIVITY_NAME}")
                    当前项目.back_to_index()

                if len(当前项目.任务列表) > 0 and 当前项目.任务列表[0].type.split('_')[0] == 'maa':
                    当前项目.MAA任务调度器((当前项目.任务列表[0].type.split('_')[1]).split(','), 首次=True)
                    continue
                当前项目.run()
                重连次数 = 0
            except ConnectionError as e:
                重连次数 += 1
                if 重连次数 < 重连次数上限:
                    logger.warning(f'连接端口断开...正在重连...')
                    连接状态 = False
                    while not 连接状态:
                        if self.stopped(): break
                        try:
                            当前项目 = 初始化([], 当前项目)
                            break
                        except Exception as ce:
                            logger.error(ce)
                            time.sleep(1)
                            continue
                    continue
                else: raise Exception(e)
            except Exception as E:
                logger.exception(f"程序出错--->{E}")

    @property
    def stop_event(self): return self._stop_event


def 终止线程报错(tid, exctype):
    """raises the exception, performs cleanup if needed"""
    tid = ctypes.c_long(tid)
    if not inspect.isclass(exctype): exctype = type(exctype)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, ctypes.py_object(exctype))
    if res == 0: raise ValueError("invalid thread id")
    elif not res == 1:
        # """if it returns a number greater than one, you're in trouble,
        # and you should call it again with exc=NULL to revert the effect"""
        ctypes.pythonapi.PyThreadState_SetAsyncExc(tid, None)
        raise SystemError("PyThreadState_SetAsyncExc failed")


def 重新运行Mower0():
    global Mower0线程, 工位表, 当前项目
    while not Mower0线程.stopped():
        try:
            # while 当前项目.MAA.running():
            #     当前项目.MAA.stop
            Mower0线程._stop_event.set()
            终止线程报错(Mower0线程.ident, SystemExit)
        except: pass
    logger.warning('Mower0已停止，准备重新启动Mower0')
    工位表 = {}
    Mower0线程 = 线程()
    Mower0线程.start()
    显示字幕()


def 停止运行Mower0():
    global Mower0线程, 当前项目
    while not Mower0线程.stopped():
        try:
            字幕窗口.withdraw()
            Mower0线程._stop_event.set()
            终止线程报错(Mower0线程.ident, SystemExit)
        except: pass
    try:
        当前项目.MAA.stop()
    except Exception:
        pass
    logger.warning('Mower0已停止')


def 退出Mower0(): os.kill(os.getpid(), 9)


def 跑单任务查询(icon: pystray.Icon): icon.notify(任务提示, "Mower0跑单任务列表")


精英化零阶段等级所需经验 = [
    0, 100, 217, 351, 502, 670, 855, 1057, 1276, 1512, 1765, 2035, 2322, 2626, 2947, 3285, 3640, 4012, 4401, 4807, 5230,
    5670, 6127, 6601, 7092, 7600, 8125, 8667, 9226, 9800, 10389, 10994, 11615, 12252, 12905, 13574, 14259, 14960, 15676,
    16400, 17139, 17888, 18647, 19417, 20200, 21004, 21824, 22660, 23512, 24400]
精英化零阶段等级所需龙门币 = [
    0, 30, 66, 109, 159, 216, 281, 354, 435, 525, 624, 732, 850, 978, 1116, 1265, 1425, 1607, 1813, 2044, 2302, 2588,
    2903, 3249, 3627, 4038, 4484, 4966, 5486, 6043, 6638, 7273, 7950, 8670, 9434, 10243, 11099, 12003, 12955, 13947,
    14989, 16075, 17206, 18384, 19613, 20907, 22260, 23673, 25147, 26719]
精英化一阶段等级所需经验 = [
    0, 120, 292, 516, 792, 1120, 1500, 1932, 2416, 2952, 3540, 4180, 4872, 5616, 6412, 7260, 8160, 9112, 10116, 11172,
    12280, 13440, 14652, 15916, 17232, 18600, 20020, 21492, 23016, 24592, 26220, 27926, 29710, 31572, 33512, 35530,
    37626, 39800, 42052, 44382, 46790, 49374, 52134, 55070, 58182, 61470, 64934, 68574, 72390, 76382, 80550, 84894,
    89414, 94110, 99000, 104326, 110345, 116657, 123162, 130000, 137391, 145048, 152871, 160960, 169315, 177936, 186823,
    195976, 205395, 215000, 224951, 235399, 246344, 257786, 269725, 282161, 295094, 308524, 322451, 337000]
精英化一阶段等级所需龙门币 = [
    0, 48, 119, 214, 334, 480, 653, 854, 1085, 1347, 1640, 1966, 2327, 2723, 3155, 3625, 4133, 4681, 5270, 5901, 6576,
    7295, 8060, 8871, 9730, 10638, 11596, 12606, 13668, 14784, 15955, 17200, 18522, 19922, 21402, 22964, 24609, 26340,
    28157, 30063, 32059, 34230, 36579, 39110, 41827, 44734, 47834, 51132, 54631, 58336, 62250, 66377, 70721, 75286,
    80093, 85387, 91436, 97849, 104530, 111628, 119381, 127497, 135875, 144627, 153759, 163277, 173186, 183492, 194201,
    205228, 216761, 228985, 241911, 255550, 269913, 285010, 300853, 317452, 334819, 353122]
精英化二阶段等级所需经验 = [
    0, 191, 494, 909, 1436, 2075, 2826, 3689, 4664, 5751, 6950, 8261, 9684, 11219, 12866, 14625, 16496, 18479, 20574,
    22781, 25100, 27531, 30074, 32729, 35496, 38375, 41366, 44469, 47684, 51011, 54450, 58052, 61817, 65745, 69836,
    74090, 78507, 83087, 87830, 92736, 97805, 103037, 108432, 113990, 119711, 125595, 131642, 137852, 144225, 150761,
    157460, 164362, 171467, 178775, 186286, 194000, 201917, 210037, 218360, 226886, 235615, 244778, 254375, 264406,
    274871, 285770, 297103, 308870, 321071, 333800, 346869, 360616, 375041, 390144, 405925, 422384, 439521, 457336,
    475829, 495000, 514849, 535954, 558315, 581932, 606805, 632934, 660319, 688960, 718857, 750000]
精英化二阶段等级所需龙门币 = [
    0, 76, 200, 373, 598, 877, 1211, 1603, 2054, 2567, 3144, 3786, 4496, 5276, 6127, 7052, 8053, 9132, 10291, 11531,
    12855, 14265, 15763, 17351, 19031, 20804, 22673, 24640, 26707, 28876, 31149, 33562, 36118, 38820, 41671, 44674,
    47832, 51148, 54625, 58265, 62072, 66048, 70197, 74521, 79023, 83707, 88575, 93630, 98875, 104313, 109947, 115814,
    121917, 128260, 134847, 141682, 148768, 156108, 163707, 171568, 179695, 188308, 197416, 207026, 217146, 227783,
    238946, 250642, 262880, 275762, 289105, 303264, 318252, 334080, 350761, 368306, 386728, 406039, 426252, 447378,
    469470, 493192, 518572, 545637, 574415, 604934, 637221, 671304, 707210, 744955]

header = {
    'cred': '',
    'User-Agent': 'Skland/1.0.1 (com.hypergryph.skland; build:100001014; Android 31; ) Okhttp/4.11.0',
    'Accept-Encoding': 'gzip',
    'Connection': 'close'
}
header_login = {
    'User-Agent': 'Skland/1.0.1 (com.hypergryph.skland; build:100001014; Android 31; ) Okhttp/4.11.0',
    'Accept-Encoding': 'gzip',
    'Connection': 'close'
}

# 签名请求头一定要这个顺序，否则失败
# timestamp是必填的,其它三个随便填,不要为none即可
header_for_sign = {
    'platform': '',
    'timestamp': '',
    'dId': '',
    'vName': ''
}


def generate_signature(token: str, path, body_or_query):
    """
    获得签名头
    接口地址+方法为Get请求？用query否则用body+时间戳+ 请求头的四个重要参数（dId，platform，timestamp，vName）.toJSON()
    将此字符串做HMAC加密，算法为SHA-256，密钥token为请求cred接口会返回的一个token值
    再将加密后的字符串做MD5即得到sign
    :param token: 拿cred时候的token
    :param path: 请求路径（不包括网址）
    :param body_or_query: 如果是GET，则是它的query。POST则为它的body
    :return: 计算完毕的sign
    """
    # 总是说请勿修改设备时间，怕不是yj你的服务器有问题吧，所以这里特地-2
    t = str(int(time.time()) - 2)
    token = token.encode('utf-8')
    header_ca = json.loads(json.dumps(header_for_sign))
    header_ca['timestamp'] = t
    header_ca_str = json.dumps(header_ca, separators=(',', ':'))
    s = path + body_or_query + t + header_ca_str
    hex_s = hmac.new(token, s.encode('utf-8'), hashlib.sha256).hexdigest()
    md5 = hashlib.md5(hex_s.encode('utf-8')).hexdigest().encode('utf-8').decode('utf-8')  # 算出签名
    return md5, header_ca


def get_sign_header(url: str, method, body, old_header, sign_token):
    h = json.loads(json.dumps(old_header))
    p = parse.urlparse(url)
    if method.lower() == 'get':
        h['sign'], header_ca = generate_signature(sign_token, p.path, p.query)
    else:
        h['sign'], header_ca = generate_signature(sign_token, p.path, json.dumps(body))
    for i in header_ca: h[i] = header_ca[i]
    return h


def login_by_password():
    r = requests.post("https://as.hypergryph.com/user/auth/v1/token_by_phone_password",
                      json={"phone": 用户配置['手机号'], "password": 用户配置['密码']}, headers=header_login).json()
    return get_token(r)


def get_cred_by_token(token):
    grant_code = get_grant_code(token)
    return get_cred(grant_code)


def get_token(resp):
    if not resp.get('status') == 0: raise Exception(f'获得token失败：{resp["msg"]}')
    return resp['data']['token']


def get_grant_code(token):
    response = requests.post("https://as.hypergryph.com/user/oauth2/v2/grant", json={
        'appCode': '4ca99fa6b56cc2ba',
        'token': token,
        'type': 0
    }, headers=header_login)
    resp = response.json()
    if not response.status_code == 200: raise Exception(f'获得认证代码失败：{resp}')
    if not resp.get('status') == 0: raise Exception(f'获得认证代码失败：{resp["msg"]}')
    return resp['data']['code']


def get_cred(grant):
    resp = requests.post("https://zonai.skland.com/api/v1/user/auth/generate_cred_by_code", json={
        'code': grant,
        'kind': 1
    }, headers=header_login).json()
    if not resp['code'] == 0: raise Exception(f'获得cred失败：{resp["message"]}')
    return resp['data']


def get_binding_list(sign_token):
    v = []
    resp = requests.get("https://zonai.skland.com/api/v1/game/player/binding",
                        headers=get_sign_header("https://zonai.skland.com/api/v1/game/player/binding",
                                                'get', None, header, sign_token)).json()

    if not resp['code'] == 0:
        logger.warning(f"请求角色列表出现问题：{resp['message']}")
        if resp.get('message') == '用户未登录':
            logger.warning(f'用户登录可能失效了，请重新运行此程序！')
            return []
    for i in resp['data']['list']:
        if not i.get('appCode') == 'arknights': continue
        v.extend(i.get('bindingList'))
    return v


def 森空岛签到():
    global 已签到日期
    try:
        if 用户配置['登录凭证'] == '否':
            登录凭证 = login_by_password()
        else:
            登录凭证 = 用户配置['登录凭证']
        sign_token = get_cred_by_token(登录凭证)['token']
        header['cred'] = get_cred_by_token(登录凭证)['cred']
        characters = get_binding_list(sign_token)
        for i in characters:
            body = {
                'gameId': 1,
                'uid': i.get('uid')
            }
            resp = requests.post("https://zonai.skland.com/api/v1/game/attendance",
                                 headers=get_sign_header("https://zonai.skland.com/api/v1/game/attendance",
                                                         'post', body, header, sign_token), json=body).json()
            if resp['code'] == 0:
                已签到日期 = datetime.now().strftime('%Y年%m月%d日')
                logger.warning(f'今天是{已签到日期}，{i.get("nickName")}({i.get("channelName")})在森空岛签到成功！')
                for j in resp['data']['awards']:
                    res = j['resource']
                    logger.warning(f'获得了{res["name"]} × {j.get("count") or 1}')
                    if 弹窗提醒: 托盘图标.notify(
                        f'{i.get("nickName")}({i.get("channelName")})在森空岛签到成功！\n获得了{res["name"]} × {j.get("count") or 1}',
                        "森空岛签到")
            elif resp['code'] == 10001:
                已签到日期 = datetime.now().strftime('%Y年%m月%d日')
                logger.info(f'今天是{已签到日期}，{i.get("nickName")}({i.get("channelName")})今天在森空岛已经签过到了')
            else:
                logger.warning(f'{i.get("nickName")}({i.get("channelName")})签到失败了！原因：{resp.get("message")}')
                if 弹窗提醒:
                    托盘图标.notify(
                        f'{i.get("nickName")}({i.get("channelName")})签到失败了！原因：{resp.get("message")}', 森空岛签到)
                continue
    except Exception as ex:
        logger.warning(f'森空岛签到失败，原因：{str(ex)}')
        logger.error('', exc_info=ex)


def 森空岛获取信息():
    try:
        if 用户配置['登录凭证'] == '否':
            登录凭证 = login_by_password()
        else:
            登录凭证 = 用户配置['登录凭证']
        森空岛小秘书角色UID = str(用户配置.get('森空岛小秘书角色UID'))  # character['uid'] 是 str，把 森空岛小秘书角色UID 转成 str
        sign_token = get_cred_by_token(登录凭证)['token']
        header['cred'] = get_cred_by_token(登录凭证)['cred']
        characters = get_binding_list(sign_token)
        if any(character.get('uid') == 森空岛小秘书角色UID for character in characters):
            # 如果确实绑定了 用户配置['森空岛小秘书角色UID']这个角色，就使用这个账号
            uid = 森空岛小秘书角色UID
        elif characters:
            uid = characters[0].get('uid')  # 否则就用第一个角色
        else:
            return  # 如果没有角色，就不获取信息了
        url = f"https://zonai.skland.com/api/v1/game/player/info?uid={uid}"
        headers = get_sign_header(url, 'get', None, header, sign_token)
        内容 = requests.get(url, headers=headers).json()
        with open('森空岛数据.json', 'w', encoding='utf-8') as 保存:
            json.dump(内容, 保存)
    except Exception as e:
        logger.warning(f'森空岛信息获取失败，原因：{e!r}')


_理智回满剩余时间 = 9999
_公开招募刷新次数 = 0
_无人机充满剩余时间 = 9999
_可赠线索 = 0
_线索交流剩余时间 = 9999
_干员心情提示名单 = str()
_信息内容 = str()


def 森空岛实时数据分析():
    森空岛获取信息()
    门牌号 = {
        'slot_3': 'B401',
        'slot_5': 'B301',
        'slot_6': 'B302',
        'slot_7': 'B303',
        'slot_9': 'B304',
        'slot_13': 'B305',
        'slot_14': 'B201',
        'slot_15': 'B202',
        'slot_16': 'B203',
        'slot_20': 'B204',
        'slot_23': 'B205',
        'slot_24': 'B101',
        'slot_25': 'B102',
        'slot_26': 'B103',
        'slot_28': 'B104',
        'slot_32': 'B105',
        'slot_34': '控制中枢',
        'slot_36': '1F02',
    }

    global _理智回满剩余时间, _公开招募刷新次数, _无人机充满剩余时间, _可赠线索, _线索交流剩余时间, _干员心情提示名单, _信息内容
    数据 = json.load(open('森空岛数据.json'))['data']
    提示 = False
    提示信息 = str()
    信息内容 = str()

    # 理智
    当前理智 = 数据['status']['ap']['current'] + (数据['currentTs'] - 数据['status']['ap']['lastApAddTime']) / 360
    理智回满剩余时间 = 数据['status']['ap']['completeRecoveryTime'] - 数据['currentTs']  # 秒
    if 理智回满剩余时间 < 0:
        当前理智 = 数据['status']['ap']['max']
        理智回满剩余时间 = 0
    if 理智回满剩余时间 < 3600:
        信息内容 += f"当前理智 {int(当前理智)}"
        if 理智回满剩余时间 > 0: 信息内容 += f"，距离理智回满前还有 {理智回满剩余时间 // 3600}小时{理智回满剩余时间 % 3600 // 60}分钟"
        信息内容 += "\n"
        提示信息 += f"当前理智 {int(当前理智)}！\n"
        if _理智回满剩余时间 >= 3600: 提示 = True
    _理智回满剩余时间 = 理智回满剩余时间

    # 公开招募刷新
    公开招募刷新次数 = 数据['building']['hire']['refreshCount']
    公开招募刷新次数填充时间 = 数据['building']['hire']['completeWorkTime'] - 数据['currentTs']  # 秒
    if 公开招募刷新次数填充时间 < 0: 公开招募刷新次数 += 1
    if 公开招募刷新次数 > 3: 公开招募刷新次数 = 3

    if 公开招募刷新次数 >= 2:
        信息内容 += f"当前公开招募可刷新{公开招募刷新次数}次"
        if 公开招募刷新次数 < 3 and 0 < 公开招募刷新次数填充时间 < 3600:
            信息内容 += f"，{公开招募刷新次数填充时间 // 60}分钟后会填充次数"
        信息内容 += "\n"
        提示信息 += f"当前公开招募可刷新{公开招募刷新次数}次\n"
        if _公开招募刷新次数 < 公开招募刷新次数 or 公开招募刷新次数 == 3: 提示 = True
    _公开招募刷新次数 = 公开招募刷新次数

    # 无人机
    当前无人机 = (数据['building']['labor']['value']
                  + (数据['building']['labor']['maxValue'] - 数据['building']['labor']['value'])
                  * (数据['currentTs'] - 数据['building']['labor']['lastUpdateTime'])
                  / 数据['building']['labor']['remainSecs'])
    无人机充满剩余时间 = (数据['building']['labor']['lastUpdateTime']
                          + 数据['building']['labor']['remainSecs'] - 数据['currentTs'])
    if 无人机充满剩余时间 < 0:
        当前无人机 = 数据['building']['labor']['maxValue']
        无人机充满剩余时间 = 0
    if 无人机充满剩余时间 < 3600:
        信息内容 += f"当前无人机 {int(当前无人机)}个，{无人机充满剩余时间 // 3600}小时{无人机充满剩余时间 % 3600 // 60}分钟后充满\n"
        提示信息 += f"当前无人机 {int(当前无人机)}个！\n"
        if _无人机充满剩余时间 >= 3600: 提示 = True
    _无人机充满剩余时间 = 无人机充满剩余时间

    # 线索交流
    可赠线索 = 数据['building']['meeting']['clue']['own']
    线索交流剩余时间 = max(数据['building']['meeting']['clue']['shareCompleteTime'] - 数据['currentTs'], 0)
    if 可赠线索 > 8:
        信息内容 += f"可赠线索{可赠线索}个\n"
        提示信息 += f"可赠线索{可赠线索}个！\n"
        if _可赠线索 < 可赠线索: 提示 = True
    if 线索交流剩余时间 < 60:
        信息内容 += f"距线索交流结束还有 {线索交流剩余时间 // 3600}小时{线索交流剩余时间 % 3600 // 60}分钟\n"
        提示信息 += f"距线索交流结束还有 {线索交流剩余时间 // 3600}小时{线索交流剩余时间 % 3600 // 60}分钟！\n"
        if _线索交流剩余时间 >= 60:   提示 = True
    _可赠线索 = 可赠线索
    _线索交流剩余时间 = 线索交流剩余时间

    # 贸易站缺人
    贸易站缺人 = False
    for 贸易站 in 数据['building']['tradings']:
        if len(贸易站['chars']) < 贸易站['level']:
            if not 贸易站缺人:
                提示 = True
                贸易站缺人 = True
    if 贸易站缺人: 提示信息 += "贸易站缺人！\n"

    # 制造站缺人
    制造站缺人 = False
    for 制造站 in 数据['building']['manufactures']:
        if len(制造站['chars']) < 制造站['level']:
            if not 制造站缺人:
                提示 = True
                制造站缺人 = True
    if 制造站缺人: 提示信息 += "制造站缺人！\n"

    # 干员心情
    干员心情提示名单 = str()
    疲劳干员名单 = str()
    for 疲劳干员 in 数据['building']['tiredChars']:
        干员心情提示名单 += 疲劳干员['charId']
        if not 疲劳干员名单 == '': 疲劳干员名单 += '、'
        if 疲劳干员['charId'] == 'char_285_medic2':
            承曦格雷伊 = False
            在发电站 = False
            for 房间 in 数据['building']['powers']:
                for 干员 in 房间['chars']:
                    if 干员['charId'] == 'char_1027_greyy2':
                        承曦格雷伊 = True
                        break
                    if 干员['charId'] == 'char_285_medic2': 在发电站 = True
            if 承曦格雷伊 or not 在发电站: continue
        疲劳干员名单 += 数据['charInfoMap'][疲劳干员['charId']]['name']
    if not 疲劳干员名单 == str():
        信息内容 += f"疲劳干员：{疲劳干员名单} \n"
        提示信息 += f"疲劳干员：{疲劳干员名单} \n"

    感知信息 = False
    人间烟火 = False
    for 基建类目 in 数据['building']:
        if 基建类目 in ['powers', 'manufactures', 'tradings', 'dormitories']:
            for 房间 in 数据['building'][基建类目]:
                for 干员 in 房间['chars']:
                    # 跳过心情意义不大的干员
                    if 数据['charInfoMap'][干员['charId']]['name'] in [
                        '纯烬艾雅法拉', '杜林', '夜莺', '凛冬', '刺玫', '流明', '波登可', '桃金娘', '爱丽丝', '四月', '闪灵',
                        '车尔尼', '寒檀', '特米米', '黑', '初雪', '临光', '冰酿', '塑心', ]: continue
                    if 干员['charId'] == 'char_391_rosmon': 感知信息 = True
                    if 干员['charId'] == 'char_455_nothin': 人间烟火 = True
                    if len(数据['building']['powers']) < 3:
                        if 数据['charInfoMap'][干员['charId']]['name'] in ['至简', ]: continue
                    if 数据['building']['hire']['level'] == 1:
                        if 数据['charInfoMap'][干员['charId']]['name'] in ['桑葚', '絮雨', '琴柳', ]: continue
                    干员心情 = 干员['ap'] / 360000
                    if 基建类目 == 'powers':
                        # Lancet-2
                        if 干员['charId'] == 'char_285_medic2':
                            承曦格雷伊 = False
                            for 房间 in 数据['building']['powers']:
                                for 发电站干员 in 房间['chars']:
                                    if 发电站干员['charId'] == 'char_1027_greyy2': 承曦格雷伊 = True
                            if 承曦格雷伊: break
                    if 基建类目 == 'dormitories':
                        # 菲亚梅塔
                        if 干员['charId'] == 'char_300_phenxi':
                            干员心情 = min(干员心情 + (数据['currentTs'] - 干员['lastApAddTime']) / 1800, 24)
                        if 干员心情 > 17:
                            刺玫 = False
                            for 同宿舍干员 in 房间['chars']:
                                if 同宿舍干员['charId'] == 'char_494_vendla':
                                    刺玫 = True
                                    break
                            if 刺玫:
                                干员心情提示名单 += 干员['charId']
                                信息内容 += f"{数据['charInfoMap'][干员['charId']]['name']}在刺玫的宿舍 {门牌号[房间['slotId']]} 心情达到了 {round(干员心情, 2)}\n"
                                提示信息 += f"{数据['charInfoMap'][干员['charId']]['name']}在刺玫的宿舍 {门牌号[房间['slotId']]} 心情达{round(干员心情, 2)}！\n"
                            elif 干员心情 > 23.5:
                                干员心情提示名单 += 干员['charId']
                                信息内容 += f"{数据['charInfoMap'][干员['charId']]['name']}的心情达到了{round(干员心情, 2)}\n"
                                提示信息 += f"{数据['charInfoMap'][干员['charId']]['name']}的心情达{round(干员心情, 2)}！\n"
                        elif 干员['charId'] == 'char_2023_ling':
                            if 感知信息 and 11.8 < 干员心情 < 18:
                                干员心情提示名单 += 'char_2023_ling'
                                信息内容 += f"令的心情达到了{round(干员心情, 2)}\n"
                                提示信息 += f"令的心情达{round(干员心情, 2)}！\n"
                        elif 干员['charId'] == 'char_2015_dusk':
                            if 人间烟火 and 11.8 < 干员心情 < 18:
                                干员心情提示名单 += 'char_2015_dusk'
                                信息内容 += f"夕的心情达到了{round(干员心情, 2)}\n"
                                提示信息 += f"夕的心情达{round(干员心情, 2)}！\n"
                    elif 干员心情 < 1:
                        干员心情提示名单 += 干员['charId']
                        信息内容 += f"{数据['charInfoMap'][干员['charId']]['name']}的心情仅剩{round(干员心情, 2)}\n"
                        提示信息 += f"{数据['charInfoMap'][干员['charId']]['name']}的心情仅剩{round(干员心情, 2)}了！\n"
        elif 基建类目 in ['control', 'meeting', 'hire']:
            for 干员 in 数据['building'][基建类目]['chars']:
                干员心情 = 干员['ap'] / 360000
                if 干员心情 < 1:
                    干员心情提示名单 += 干员['charId']
                    信息内容 += f"{数据['charInfoMap'][干员['charId']]['name']}的心情仅剩{round(干员心情, 2)}\n"
                    提示信息 += f"{数据['charInfoMap'][干员['charId']]['name']}的心情仅剩{round(干员心情, 2)}了！\n"
                elif 干员['charId'] == 'char_2023_ling':
                    if 人间烟火 and 11 < 干员心情 < 12.1:
                        干员心情提示名单 += 'char_2023_ling'
                        信息内容 += f"令的心情仅剩{round(干员心情, 2)}\n"
                        提示信息 += f"令的心情仅剩{round(干员心情, 2)}！\n"
                elif 干员['charId'] == 'char_2015_dusk':
                    if 感知信息 and 6 < 干员心情 < 12.1:
                        干员心情提示名单 += 'char_2015_dusk'
                        信息内容 += f"夕的心情仅剩{round(干员心情, 2)}\n"
                        提示信息 += f"夕的心情仅剩{round(干员心情, 2)}！\n"

    # 贸易站接单时间
    # for 贸易站 in 数据['building']['tradings']: if 贸易站['completeWorkTime'] > 数据['currentTs']: 信息内容 += f"贸易站 {门牌号[贸易站['slotId']]} 接单时间为 {datetime.fromtimestamp(贸易站['completeWorkTime']).strftime('%H:%M:%S')}\n"

    if not _干员心情提示名单 == 干员心情提示名单:
        提示 = True
        _干员心情提示名单 = 干员心情提示名单

    if 信息内容 == str(): 信息内容 = "目前没有亟待解决的问题。\n"

    输出信息 = f"\n{数据['status']['name'].split('#')[0]} 博士，\n按照森空岛的目前采集到的信息推算，罗德岛需要注意的情况向您汇报如下：\n"
    输出信息 += 信息内容
    输出信息 += (
        "------------------------------------\n森空岛信息刷新时间 "
        f"{datetime.fromtimestamp(数据['building']['labor']['lastUpdateTime']).strftime('%Y-%m-%d %H:%M')}\n"
        f"森空岛本次汇报时间 {datetime.fromtimestamp(数据['currentTs']).strftime('%Y-%m-%d %H:%M')}\n")

    已输出 = False
    if not 信息内容 == _信息内容:
        if not 信息内容 == "目前没有亟待解决的问题。":
            print(输出信息)
            运行信息滚动窗.insert(END, 输出信息 + '\n')
            运行信息滚动窗.yview(END)  # 自动滚动到底部
            已输出 = True
        _信息内容 = 信息内容
    if 提示:
        if len(提示信息) > 256: 提示信息 = "问题过多，求求你上游戏自己看看吧！"
        托盘图标.notify(提示信息, '森空岛小秘书')
    return 输出信息, 已输出


def 森空岛查看游戏内信息():
    输出信息, 已输出 = 森空岛实时数据分析()
    if not 已输出:
        print(输出信息)
        运行信息滚动窗.insert(END, 输出信息 + '\n')
        运行信息滚动窗.yview(END)  # 自动滚动到底部


def 森空岛干员阵容查询():
    try: 森空岛获取信息()
    except Exception as ex: logger.warning(f'森空岛信息获取失败，原因：{str(ex)}')
    数据 = json.load(open('森空岛数据.json'))['data']
    阵容内容 = f"\n{数据['status']['name'].split('#')[0]} 博士，根据从森空岛采集到的信息，罗德岛目前的阵容概况如下：\n"
    总计消耗经验 = 0
    总计消耗龙门币 = 0
    with open('森空岛干员阵容查询.csv', 'w', newline="") as 干员阵容文件:
        导出干员阵容 = csv.writer(干员阵容文件)
        导出干员阵容.writerow(
            ['干员代号', '潜能', '精英化阶段', '干员等级', '技能等级', '技能1专精等级', '技能2专精等级',
             '技能3专精等级', '模组',
             '消耗龙门币', '消耗经验', '消耗龙门币/经验'])
        # 计算干员精英化与升级的经验和龙门币花销
        for 干员 in 数据['chars']:
            导出干员信息 = []
            阵容内容 += "\n"
            干员代号 = 数据['charInfoMap'][干员['charId']]['name']
            精英化阶段 = '零'
            消耗经验 = 0
            消耗龙门币 = 0
            if 干员['evolvePhase'] == 0:
                消耗经验 = 精英化零阶段等级所需经验[干员['level'] - 1]
                消耗龙门币 = 精英化零阶段等级所需龙门币[干员['level'] - 1]
            elif 干员['evolvePhase'] == 1:
                精英化阶段 = '一'
                if 数据['charInfoMap'][干员['charId']]['rarity'] == 2:
                    消耗经验 = 16400 + 精英化一阶段等级所需经验[干员['level'] - 1]
                    消耗龙门币 = 23947 + 精英化一阶段等级所需龙门币[干员['level'] - 1]
                elif 数据['charInfoMap'][干员['charId']]['rarity'] == 3:
                    消耗经验 = 20200 + 精英化一阶段等级所需经验[干员['level'] - 1]
                    消耗龙门币 = 34613 + 精英化一阶段等级所需龙门币[干员['level'] - 1]
                elif 数据['charInfoMap'][干员['charId']]['rarity'] == 4:
                    消耗经验 = 24400 + 精英化一阶段等级所需经验[干员['level'] - 1]
                    消耗龙门币 = 46719 + 精英化一阶段等级所需龙门币[干员['level'] - 1]
                elif 数据['charInfoMap'][干员['charId']]['rarity'] == 5:
                    消耗经验 = 24400 + 精英化一阶段等级所需经验[干员['level'] - 1]
                    消耗龙门币 = 56719 + 精英化一阶段等级所需龙门币[干员['level'] - 1]
            elif 干员['evolvePhase'] == 2:
                精英化阶段 = '二'
                if 数据['charInfoMap'][干员['charId']]['rarity'] == 3:
                    消耗经验 = 150200 + 精英化二阶段等级所需经验[干员['level'] - 1]
                    消耗龙门币 = 206241 + 精英化二阶段等级所需龙门币[干员['level'] - 1]
                elif 数据['charInfoMap'][干员['charId']]['rarity'] == 4:
                    消耗经验 = 239400 + 精英化二阶段等级所需经验[干员['level'] - 1]
                    消耗龙门币 = 371947 + 精英化二阶段等级所需龙门币[干员['level'] - 1]
                elif 数据['charInfoMap'][干员['charId']]['rarity'] == 5:
                    消耗经验 = 361400 + 精英化二阶段等级所需经验[干员['level'] - 1]
                    消耗龙门币 = 589841 + 精英化二阶段等级所需龙门币[干员['level'] - 1]
            if 干员['charId'] == 'char_1001_amiya2':
                消耗经验 = 0
                消耗龙门币 = 0
                干员代号 += "-近卫"
            导出干员信息.append(干员代号)
            阵容内容 += 干员代号
            导出干员信息.append(干员['potentialRank'] + 1)
            导出干员信息.append(精英化阶段)
            导出干员信息.append(干员['level'])
            阵容内容 += f"：精英化{精英化阶段}阶段{干员['level']}级"
            导出干员信息.append(干员['mainSkillLvl'])
            for 技能 in range(3):
                if 技能 < len(干员['skills']):
                    导出干员信息.append(干员['skills'][技能]['specializeLevel'])
                else:
                    导出干员信息.append('')

            # 计算干员模组的龙门币花销
            导出模组信息 = ''
            if 干员['evolvePhase'] == 2:
                for 模组 in 干员['equip']:
                    模组是开的 = False
                    if 模组['level'] == 3:
                        模组是开的 = True
                        if 数据['charInfoMap'][干员['charId']]['rarity'] == 3: 消耗龙门币 += 75000
                        elif 数据['charInfoMap'][干员['charId']]['rarity'] == 4: 消耗龙门币 += 150000
                        elif 数据['charInfoMap'][干员['charId']]['rarity'] == 5: 消耗龙门币 += 300000
                    elif 模组['level'] == 2:
                        模组是开的 = True
                        if 数据['charInfoMap'][干员['charId']]['rarity'] == 3: 消耗龙门币 += 45000
                        elif 数据['charInfoMap'][干员['charId']]['rarity'] == 4: 消耗龙门币 += 90000
                        elif 数据['charInfoMap'][干员['charId']]['rarity'] == 5: 消耗龙门币 += 180000
                    elif not 数据['equipmentInfoMap'][模组['id']]['typeIcon'] == 'original' and 模组['id'] == 干员[
                        'defaultEquipId']:
                        模组是开的 = True
                        if 数据['charInfoMap'][干员['charId']]['rarity'] == 3: 消耗龙门币 += 20000
                        elif 数据['charInfoMap'][干员['charId']]['rarity'] == 4: 消耗龙门币 += 40000
                        elif 数据['charInfoMap'][干员['charId']]['rarity'] == 5: 消耗龙门币 += 80000
                    if 模组是开的:
                        if not 导出模组信息 == '': 导出模组信息 += '\n'
                        导出模组信息 += f"「{数据['equipmentInfoMap'][模组['id']]['name']}」等级{模组['level']}"
                        阵容内容 += f"，模组「{数据['equipmentInfoMap'][模组['id']]['name']}」等级{模组['level']}"
            导出干员信息.append(导出模组信息)
            导出干员信息.append(消耗龙门币)
            导出干员信息.append(消耗经验)
            if 消耗经验 == 0: 导出干员信息.append('')
            else:
                导出干员信息.append(round(消耗龙门币 / 消耗经验, 3))
                阵容内容 += f"，消耗龙门币 {消耗龙门币} / 经验 {消耗经验} = {round(消耗龙门币 / 消耗经验, 3)}"
            总计消耗经验 += 消耗经验
            总计消耗龙门币 += 消耗龙门币
            导出干员阵容.writerow(导出干员信息)
        导出干员阵容.writerow(
            ['总计', '', '', '', '', '', '', '', '', 总计消耗龙门币, 总计消耗经验, 总计消耗龙门币 / 总计消耗经验])
    print(阵容内容)
    运行信息滚动窗.insert(END, 阵容内容 + '\n\n')
    运行信息滚动窗.yview(END)  # 自动滚动到底部
    logger.warning(f"总计消耗龙门币 {总计消耗龙门币} / 经验 {总计消耗经验} = {round(总计消耗龙门币 / 总计消耗经验, 3)}")


def 增加干员(贸易站序号: int):
    位置 = len(跑单位置表[贸易站序号]) - 3
    跑单位置表[贸易站序号][位置] = [界面.StringVar(value=''), 界面.Label(跑单位置表[贸易站序号]["房间"], image=不跑单图标)]
    跑单位置表[贸易站序号][位置][1].grid(row=1, column=位置, padx=10, pady=5, sticky=界面.W)
    界面.Radiobutton(跑单位置表[贸易站序号]["房间"], text='不跑单', variable=跑单位置表[贸易站序号][位置][0], value='').grid(row=2, column=位置, padx=10, pady=5, sticky=界面.W)
    界面.Radiobutton(跑单位置表[贸易站序号]["房间"], text='但书', variable=跑单位置表[贸易站序号][位置][0], value='但书').grid(row=3, column=位置, padx=10, pady=5, sticky=界面.W)
    界面.Radiobutton(跑单位置表[贸易站序号]["房间"], text='龙舌兰', variable=跑单位置表[贸易站序号][位置][0], value='龙舌兰').grid(row=4, column=位置, padx=10, pady=5, sticky=界面.W)
    if 位置 == 2:
        跑单位置表[贸易站序号]["+"].destroy()
        del 跑单位置表[贸易站序号]["+"]


def 增加贸易站():
    跑单位置表.append({"房间": 界面.Frame(跑单位置设置, relief=界面.RIDGE, borderwidth=10)})
    跑单位置表[-1]["房间"].grid(row=len(跑单位置表)-1, column=0, columnspan=2, padx=5, pady=5, sticky=界面.W+E+N+S)
    界面.Label(跑单位置表[-1]["房间"], text="门牌号").grid(row=0, column=0, padx=10, pady=5, sticky=界面.E)
    跑单位置表[-1]["门牌号"] = 界面.Entry(跑单位置表[-1]["房间"], justify=LEFT, width=16)
    跑单位置表[-1]["门牌号"].grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky=界面.W)
    跑单位置表[-1][0] = [界面.StringVar(value=''), 界面.Label(跑单位置表[-1]["房间"], image=不跑单图标)]
    界面.Radiobutton(跑单位置表[-1]["房间"], text='不跑单', variable=跑单位置表[-1][0][0], value='').grid(row=2, column=0, padx=10, pady=5, sticky=界面.W)
    界面.Radiobutton(跑单位置表[-1]["房间"], text='但书', variable=跑单位置表[-1][0][0], value='但书').grid(row=3, column=0, padx=10, pady=5, sticky=界面.W)
    界面.Radiobutton(跑单位置表[-1]["房间"], text='龙舌兰', variable=跑单位置表[-1][0][0], value='龙舌兰').grid(row=4, column=0, padx=10, pady=5, sticky=界面.W)
    跑单位置表[-1][0][1].grid(row=1, column=0, padx=10, pady=5, sticky=界面.W)
    跑单位置表[-1]["+"] = 界面.Button(跑单位置表[-1]["房间"], text="+", bootstyle=(PRIMARY, "outline-toolbutton"), command=lambda 贸易站序号=len(跑单位置表)-1: 增加干员(贸易站序号), width=5)
    跑单位置表[-1]["+"].grid(row=1, column=3, padx=10, pady=5, sticky=界面.N+S)


def 减少贸易站():
    跑单位置表[-1]["房间"].destroy()
    跑单位置表.pop()


def 刷新跑单位置设置():
    for 贸易站序号, 贸易站 in enumerate(跑单位置表):
        for 序号, 项目 in enumerate(跑单位置表[贸易站序号]):
            if 项目 in [0, 1, 2]:
                if 跑单位置表[贸易站序号][项目][0].get() == '但书': 跑单干员图标 = 但书图标
                elif 跑单位置表[贸易站序号][项目][0].get() == '龙舌兰': 跑单干员图标 = 龙舌兰图标
                else: 跑单干员图标 = 不跑单图标
                跑单位置表[贸易站序号][项目][1].configure(image=跑单干员图标)
    窗口.after(100, 刷新跑单位置设置)


def 保存配置():
    try:
        用户配置['服务器'] = 服务器输入.get()
        用户配置['adb地址'] = adb地址输入.get()
        用户配置['跑单位置设置'] = {}
        for 序号, 贸易站序号 in enumerate(跑单位置表):
            贸易站 = 跑单位置表[序号]["门牌号"].get()
            if 贸易站 == '':
                logger.error("请在所有设置填写完整后再保存配置")
                return
            用户配置['跑单位置设置'][贸易站] = []
            for 项目 in 跑单位置表[序号]:
                if 项目 in [0, 1, 2]: 用户配置['跑单位置设置'][贸易站].append(跑单位置表[序号][项目][0].get())
        用户配置['日志存储目录'] = 日志存储目录输入.get()
        用户配置['截图存储目录'] = 截图存储目录输入.get()
        用户配置['每种截图的最大保存数量'] = 每种截图的最大保存数量输入.get()
        用户配置['弹窗提醒开关'] = '开' if 弹窗提醒开关输入.get() else '关'
        用户配置['悬浮字幕开关'] = '开' if 悬浮字幕开关输入.get() else '关'
        用户配置['字幕字号'] = 悬浮字幕字号输入.get()
        用户配置['字幕字体'] = 悬浮字幕字体输入.get()
        用户配置['字幕颜色'] = 悬浮字幕颜色输入.get()
        用户配置['邮件设置']['邮件提醒开关'] = '开' if 邮件提醒开关输入.get() else '关'
        用户配置['邮件设置']['发信邮箱'] = 发信邮箱输入.get()
        用户配置['邮件设置']['授权码'] = 授权码输入.get()
        用户配置['邮件设置']['收件人邮箱'] = [收件人邮箱输入.get()]
        用户配置['森空岛签到开关'] = '开' if 森空岛签到开关输入.get() else '关'
        用户配置['森空岛小秘书开关'] = '开' if 森空岛小秘书开关输入.get() else '关'
        用户配置['登录凭证'] = 登录凭证输入.get()
        用户配置['手机号'] = 手机号输入.get()
        用户配置['密码'] = 密码输入.get()
        用户配置['跑单提前运行时间'] = int(跑单提前运行时间输入.get())
        用户配置['更换干员前缓冲时间'] = int(更换干员前缓冲时间输入.get())
        用户配置['跑单消耗无人机开关'] = '开' if 跑单消耗无人机开关输入.get() else '关'
        用户配置['任务结束后退出游戏'] = '是' if 任务结束后退出游戏开关输入.get() else '否'
        用户配置['龙舌兰和但书休息'] = '开' if 龙舌兰和但书自动休息开关输入.get() else '关'
        用户配置['宿舍设置'][宿舍门牌号输入.get()] = []
        for 序号 in range(5):
            用户配置['宿舍设置'][宿舍门牌号输入.get()].append(宿舍干员安排[序号].get())
        用户配置['MAA设置']['作战开关'] = '开' if MAA作战开关输入.get() else '关'
        用户配置['MAA设置']['MAA路径'] = MAA路径输入.get()
        用户配置['MAA设置']['MAA_adb路径'] = MAA_adb路径输入.get()
        用户配置['MAA设置']['集成战略'] = '开' if 集成战略开关输入.get() else '关'
        用户配置['MAA设置']['集成战略主题'] = 集成战略主题输入.get()
        用户配置['MAA设置']['集成战略分队'] = 集成战略分队输入.get()
        用户配置['MAA设置']['集成战略开局招募组合'] = 集成战略开局招募组合输入.get()
        用户配置['MAA设置']['集成战略策略模式'] = 集成战略策略模式输入.get()
        用户配置['MAA设置']['消耗理智关卡'] = 消耗理智关卡输入.get()
        用户配置['MAA设置']['每次MAA使用理智药数量'] = 每次MAA使用理智药数量输入.get()
    except:
        logger.error("请在所有设置填写完整而合理后再保存配置")
        return
    with open('Mower0用户配置文件.yaml', 'w', encoding='utf-8') as 用户配置文件:
        yaml.dump(用户配置, 用户配置文件, allow_unicode=True)
    logger.info("配置已保存")


def 开始运行():
    global 初启动, Mower0线程, 服务器, 弹窗提醒, 跑单提前运行时间, 更换干员前缓冲时间, 龙舌兰和但书休息, 悬浮字幕开关, 签到, 森空岛小秘书, 字幕字号, 字幕颜色, 邮件设置, MAA设置
    保存配置()
    logger.info("开始运行")
    with open('Mower0用户配置文件.yaml', 'r', encoding='utf-8') as 用户配置文件:
        用户配置 = yaml.load(用户配置文件.read(), Loader=yaml.FullLoader)
    服务器 = 'com.hypergryph.arknights.bilibili' if 用户配置['服务器'] == 'Bilibili服务器' else 'com.hypergryph.arknights'
    弹窗提醒 = True if 用户配置['弹窗提醒开关'] == '开' else False
    跑单提前运行时间 = 用户配置['跑单提前运行时间']
    更换干员前缓冲时间 = 用户配置['更换干员前缓冲时间']
    龙舌兰和但书休息 = True if 用户配置['龙舌兰和但书休息'] == '开' else False
    悬浮字幕开关 = True if 用户配置['悬浮字幕开关'] == '开' else False
    签到 = True if 用户配置['森空岛签到开关'] == '开' else False
    森空岛小秘书 = True if 用户配置['森空岛小秘书开关'] == '开' else False
    if not 用户配置['字幕字号'] == '默认': 字幕字号 = int(用户配置['字幕字号'])
    字幕颜色 = 用户配置['字幕颜色']
    邮件设置 = 用户配置['邮件设置']
    MAA设置 = 用户配置['MAA设置']
    config.ADB_DEVICE = [用户配置['adb地址']]
    config.ADB_CONNECT = [用户配置['adb地址']]
    config.APPNAME = 服务器
    config.TAP_TO_LAUNCH = [{"enable": "false", "x": "0", "y": "0"}]
    if 初启动:
        Mower0线程 = 线程()
        Mower0线程.start()
        显示字幕()
        初启动 = False
    else: 重新运行Mower0()


def 运行信息滚轮(event):
    运行信息滚动窗.yview_scroll(int(-1*(event.delta/120)), "units")


def 显示字幕(): 字幕窗口.deiconify()


def 选中窗口(event):
    global 鼠标水平初始位置, 鼠标竖直初始位置

    鼠标水平初始位置 = event.x  # 获取鼠标相对于窗体左上角的X坐标
    鼠标竖直初始位置 = event.y  # 获取鼠标相对于窗左上角体的Y坐标


def 拖动窗口(event): 字幕窗口.geometry(f'+{event.x_root - 鼠标水平初始位置}+{event.y_root - 鼠标竖直初始位置}')


def 关闭窗口(icon: pystray.Icon): 字幕窗口.withdraw()


def 缩放字幕(event):
    global 字幕字号
    if event.delta > 0: 字幕字号 += 1
    else: 字幕字号 -= 1
    if 字幕字号 < 1: 字幕字号 = 1


def 更新字幕():
    global 字幕
    任务倒计时 = (下个任务开始时间 - datetime.now()).total_seconds()
    字幕 = 'Mower0的回合！'
    if 任务倒计时 > 0:
        字幕 = f'Mower0将在{int(任务倒计时 / 60)}分钟后开始跑单'
        if 任务倒计时 <= 跑单提前运行时间: 字幕 += '\n跑单即将开始！'
    悬浮字幕.config(text=字幕, font=(用户配置['字幕字体'], 字幕字号, 'bold'), bg=字幕颜色, fg=字幕颜色[:6] + str(int(字幕颜色[5] == '0')))
    字幕窗口.after(1000, 更新字幕)


初启动 = True
ocr = None
任务提示 = str()
工位表 = {}
下个任务开始时间 = datetime.now()
字幕 = str()
已签到日期 = str()
关卡掉落 = {}

if os.path.exists('Mower0用户配置文件.yaml'): 读取文件 = 'Mower0用户配置文件.yaml'
else: 读取文件 = '元素/Mower0用户配置文件 - 备份.yaml'
with open(读取文件, 'r', encoding='utf-8') as 用户配置文件:
    用户配置 = yaml.load(用户配置文件.read(), Loader=yaml.SafeLoader)
服务器 = 'com.hypergryph.arknights.bilibili' if 用户配置['服务器'] == 'Bilibili服务器' else 'com.hypergryph.arknights'
弹窗提醒 = True if 用户配置['弹窗提醒开关'] == '开' else False
跑单提前运行时间 = 用户配置['跑单提前运行时间']
更换干员前缓冲时间 = 用户配置['更换干员前缓冲时间']
龙舌兰和但书休息 = True if 用户配置['龙舌兰和但书休息'] == '开' else False
悬浮字幕开关 = True if 用户配置['悬浮字幕开关'] == '开' else False
签到 = True if 用户配置['森空岛签到开关'] == '开' else False
森空岛小秘书 = True if 用户配置['森空岛小秘书开关'] == '开' else False
if not 用户配置['字幕字号'] == '默认': 字幕字号 = int(用户配置['字幕字号'])
字幕颜色 = 用户配置['字幕颜色']
邮件设置 = 用户配置['邮件设置']
MAA设置 = 用户配置['MAA设置']

窗口 = 界面.Window(title="Mower0", themename="minty", iconphoto="元素/图标.png", size=(1550, 1000), minsize=(0, 0))
标签页集 = 界面.Notebook(窗口)
标签页集.pack(fill=BOTH, expand=YES)
标签页 = []
滚动区域 = []
for 序号 in range(3):
    标签页.append(界面.Frame(标签页集))
    标签页[序号].rowconfigure(0, weight=1)
    标签页[序号].columnconfigure(0, weight=1)
    滚动区域.append(ScrolledFrame(标签页[序号], autohide=True))
    滚动区域[序号].pack(fill=BOTH, expand=YES)
标签页集.add(标签页[0], text='主页', compound=TOP)
标签页集.add(标签页[1], text='进阶设置', compound=TOP)
标签页集.add(标签页[2], text='使用说明', compound=TOP)

# 主页
基础运行设置 = 界面.LabelFrame(滚动区域[0], text="基础运行设置", relief=界面.RIDGE, borderwidth=10)
基础运行设置.grid(row=0, column=0, sticky=界面.W+E+N+S, padx=10, pady=10)
服务器输入 = 界面.StringVar(value=用户配置['服务器'])   ###
界面.Label(基础运行设置, text="服务器").grid(row=0, column=0, padx=10, pady=5, sticky=界面.E)
界面.Radiobutton(基础运行设置, text='官方服务器', variable=服务器输入, value='官方服务器').grid(row=0, column=1, padx=5, pady=5, sticky=界面.W)
界面.Radiobutton(基础运行设置, text='Bilibili服务器', variable=服务器输入, value='Bilibili服务器').grid(row=0, column=2, padx=5, pady=5, sticky=界面.W)
界面.Label(基础运行设置, text="adb地址").grid(row=1, column=0, padx=10, pady=5, sticky=界面.E)
adb地址输入 = 界面.Entry(基础运行设置, justify=LEFT) ###
adb地址输入.grid(row=1, column=1, padx=5, pady=5, columnspan=2, sticky=界面.W+E)
adb地址输入.insert(0, 用户配置['adb地址'])

跑单位置设置 = 界面.LabelFrame(滚动区域[0], text="跑单位置设置", relief=界面.RIDGE, borderwidth=10)
跑单位置设置.grid(row=1, rowspan=5, column=0, sticky=界面.W+E+N+S, padx=10, pady=10)
不跑单图标 = ImageTk.PhotoImage(Image.open("元素/不跑单.png").resize((80, 80)))
但书图标 = ImageTk.PhotoImage(Image.open("元素/但书.png").resize((80, 80)))
龙舌兰图标 = ImageTk.PhotoImage(Image.open("元素/龙舌兰.png").resize((80, 80)))
跑单位置表 = []
for 序号, 贸易站 in enumerate(用户配置['跑单位置设置']):
    跑单位置表.append({"房间": 界面.Frame(跑单位置设置, relief=界面.RIDGE, borderwidth=10)})
    跑单位置表[序号]["房间"].grid(row=序号, column=0, columnspan=2, padx=5, pady=5, sticky=界面.W+E+N+S)
    界面.Label(跑单位置表[序号]["房间"], text="门牌号").grid(row=0, column=0, padx=10, pady=5, sticky=界面.E)
    跑单位置表[序号]["门牌号"] = 界面.Entry(跑单位置表[序号]["房间"], justify=LEFT, width=15)
    跑单位置表[序号]["门牌号"].grid(row=0, column=1, columnspan=3, padx=5, pady=5, sticky=界面.W)
    跑单位置表[序号]["门牌号"].insert(0, 贸易站)
    for 位置, 干员 in enumerate(用户配置['跑单位置设置'][贸易站]):
        跑单位置表[序号][位置] = []
        跑单位置表[序号][位置].append(界面.StringVar(value=干员))  ###
        界面.Radiobutton(跑单位置表[序号]["房间"], text='不跑单', variable=跑单位置表[序号][位置][0], value='').grid(row=2, column=位置, padx=10, pady=5, sticky=界面.W)
        界面.Radiobutton(跑单位置表[序号]["房间"], text='但书', variable=跑单位置表[序号][位置][0], value='但书').grid(row=3, column=位置, padx=10, pady=5, sticky=界面.W)
        界面.Radiobutton(跑单位置表[序号]["房间"], text='龙舌兰', variable=跑单位置表[序号][位置][0], value='龙舌兰').grid(row=4, column=位置, padx=10, pady=5, sticky=界面.W)
for 序号, 贸易站 in enumerate(跑单位置表):
    for 位置, 干员 in enumerate(用户配置['跑单位置设置'][跑单位置表[序号]["门牌号"].get()]):
        if 跑单位置表[序号][位置][0].get() == '但书': 跑单干员图标 = 但书图标
        elif 跑单位置表[序号][位置][0].get() == '龙舌兰': 跑单干员图标 = 龙舌兰图标
        else: 跑单干员图标 = 不跑单图标
        跑单位置表[序号][位置].append(界面.Label(跑单位置表[序号]["房间"], image=跑单干员图标))
        跑单位置表[序号][位置][1].grid(row=1, column=位置, padx=10, pady=5, sticky=界面.W)
    if 位置 < 2:
        跑单位置表[序号]["+"] = 界面.Button(跑单位置表[序号]["房间"], text="+", bootstyle=(PRIMARY, "outline-toolbutton"), command=lambda 贸易站序号=序号: 增加干员(贸易站序号), width=5)
        跑单位置表[序号]["+"].grid(row=1, column=3, padx=10, pady=5, sticky=界面.W+N+S)

界面.Button(跑单位置设置, text="+", bootstyle=(PRIMARY, "outline-toolbutton"), command=增加贸易站, width=14).grid(row=10, column=0, padx=5, pady=5)
界面.Button(跑单位置设置, text="-", bootstyle=(SECONDARY, "outline-toolbutton"), command=减少贸易站, width=14).grid(row=10, column=1, padx=5, pady=5)

界面.Button(滚动区域[0], text="森空岛干员阵容查询", bootstyle=(PRIMARY, "outline-toolbutton"), command=森空岛干员阵容查询, width=20).grid(row=0, column=1, padx=10, pady=23, sticky=界面.N+S)
界面.Button(滚动区域[0], text="保存配置", bootstyle=(PRIMARY, "outline-toolbutton"), command=保存配置, width=20).grid(row=0, column=2, padx=10, pady=23, sticky=界面.N+S)
界面.Button(滚动区域[0], text="开始运行", bootstyle=(PRIMARY, "outline-toolbutton"), command=开始运行, width=20).grid(row=0, column=3, padx=10, pady=23, sticky=界面.N+S)
界面.Button(滚动区域[0], text="停止运行", bootstyle=(SECONDARY, "outline-toolbutton"), command=停止运行Mower0, width=20).grid(row=0, column=4, padx=10, pady=23, sticky=界面.N+S)

界面.Separator(滚动区域[0], bootstyle="light").grid(row=1, column=1, columnspan=4, pady=5, sticky=界面.W+E)
运行信息滚动窗 = 界面.ScrolledText(滚动区域[0], width=80, height=28, wrap=界面.WORD, font=('Consolas', 11), relief=界面.RIDGE)
运行信息滚动窗.vbar.pack_forget()
运行信息滚动窗.grid(row=2, column=1, columnspan=4, sticky=界面.W+E+N, padx=10, pady=10)
运行信息滚动窗.bind("<MouseWheel>", 运行信息滚轮)

# 进阶设置
偏好运行设置 = 界面.LabelFrame(滚动区域[1], text="偏好运行设置", relief=界面.RIDGE, borderwidth=10)
偏好运行设置.grid(row=0, column=0, sticky=界面.W+E+N+S, padx=10, pady=10)
界面.Label(偏好运行设置, text="日志存储目录").grid(row=0, column=0, padx=10, pady=5, sticky=界面.W)
日志存储目录输入 = 界面.Entry(偏好运行设置, width=30, justify=LEFT)
日志存储目录输入.grid(row=0, column=1, padx=5, pady=5, columnspan=2, sticky=界面.W+E)
日志存储目录输入.insert(0, 用户配置['日志存储目录'])
界面.Label(偏好运行设置, text="截图存储目录").grid(row=1, column=0, padx=10, pady=5, sticky=界面.W)
截图存储目录输入 = 界面.Entry(偏好运行设置, width=30, justify=LEFT)    ###
截图存储目录输入.grid(row=1, column=1, padx=5, pady=5, columnspan=2, sticky=界面.W+E)
截图存储目录输入.insert(0, 用户配置['截图存储目录'])
界面.Label(偏好运行设置, text="每种截图的最大保存数量").grid(row=2, column=0, padx=10, pady=5, sticky=界面.W)
每种截图的最大保存数量输入 = 界面.Entry(偏好运行设置, width=30, justify=LEFT)    ###
每种截图的最大保存数量输入.grid(row=2, column=1, padx=5, pady=5, columnspan=2, sticky=界面.W+E)
每种截图的最大保存数量输入.insert(0, 用户配置['每种截图的最大保存数量'])

# 提醒设置
提醒设置 = 界面.LabelFrame(滚动区域[1], text="提醒设置", relief=界面.RIDGE, borderwidth=10)
提醒设置.grid(row=1, rowspan=2, column=0, sticky=界面.W+E+N+S, padx=10, pady=10)
弹窗提醒开关输入 = 界面.BooleanVar(value=False)    ###
if 用户配置['弹窗提醒开关'] == '开': 弹窗提醒开关输入.set(True)
界面.Checkbutton(提醒设置, bootstyle="round-toggle", text="弹窗提醒", variable=弹窗提醒开关输入, onvalue=TRUE, offvalue=FALSE).grid(row=0, column=0, padx=30, pady=5, sticky=界面.W)

悬浮字幕提醒 = 界面.Frame(提醒设置, relief=界面.RIDGE, borderwidth=10)
悬浮字幕提醒.grid(row=1, column=0, padx=10, pady=5, sticky=界面.W+E)
悬浮字幕开关输入 = 界面.BooleanVar(value=False)    ###
if 用户配置['悬浮字幕开关'] == '开': 悬浮字幕开关输入.set(True)
界面.Checkbutton(悬浮字幕提醒, bootstyle="round-toggle", text="悬浮字幕", variable=悬浮字幕开关输入, onvalue=TRUE, offvalue=FALSE).grid(row=0, column=0, padx=10, pady=5, sticky=界面.W)
界面.Label(悬浮字幕提醒, text="悬浮字幕字号", width=16).grid(row=1, column=0, padx=10, pady=5, sticky=界面.E)
悬浮字幕字号输入 = 界面.Entry(悬浮字幕提醒, width=30, justify=LEFT)    ###
悬浮字幕字号输入.grid(row=1, column=1, padx=5, pady=5, sticky=界面.W+E)
悬浮字幕字号输入.insert(0, 用户配置['字幕字号'])
界面.Label(悬浮字幕提醒, text="悬浮字幕字体", width=16).grid(row=2, column=0, padx=10, pady=5, sticky=界面.E)
悬浮字幕字体输入 = 界面.Entry(悬浮字幕提醒, width=30, justify=LEFT)    ###
悬浮字幕字体输入.grid(row=2, column=1, padx=5, pady=5, sticky=界面.W+E)
悬浮字幕字体输入.insert(0, 用户配置['字幕字体'])
界面.Label(悬浮字幕提醒, text="悬浮字幕颜色", width=16).grid(row=3, column=0, padx=10, pady=5, sticky=界面.E)
悬浮字幕颜色输入 = 界面.Entry(悬浮字幕提醒, width=30, justify=LEFT)    ###
悬浮字幕颜色输入.grid(row=3, column=1, padx=5, pady=5, sticky=界面.W+E)
悬浮字幕颜色输入.insert(0, 用户配置['字幕颜色'])

邮件提醒 = 界面.Frame(提醒设置, relief=界面.RIDGE, borderwidth=10)
邮件提醒.grid(row=2, column=0, padx=10, pady=5, sticky=界面.W+E)
邮件提醒开关输入 = 界面.BooleanVar(value=False)    ###
if 用户配置['邮件设置']['邮件提醒开关'] == '开': 邮件提醒开关输入.set(True)
界面.Checkbutton(邮件提醒, bootstyle="round-toggle", text="邮件提醒", variable=邮件提醒开关输入, onvalue=TRUE, offvalue=FALSE).grid(row=0, column=0, padx=10, pady=5, sticky=界面.W)
界面.Label(邮件提醒, text="发信邮箱", width=16).grid(row=1, column=0, padx=10, pady=5, sticky=界面.E)
发信邮箱输入 = 界面.Entry(邮件提醒, width=30, justify=LEFT)    ###
发信邮箱输入.grid(row=1, column=1, padx=5, pady=5, sticky=界面.W+E)
发信邮箱输入.insert(0, 用户配置['邮件设置']['发信邮箱'])
界面.Label(邮件提醒, text="授权码", width=16).grid(row=2, column=0, padx=10, pady=5, sticky=界面.E)
授权码输入 = 界面.Entry(邮件提醒, width=30, justify=LEFT, show="●")    ###
授权码输入.grid(row=2, column=1, padx=5, pady=5, sticky=界面.W+E)
授权码输入.insert(0, 用户配置['邮件设置']['授权码'])
界面.Label(邮件提醒, text="收件人邮箱", width=16).grid(row=3, column=0, padx=10, pady=5, sticky=界面.E)
收件人邮箱输入 = 界面.Entry(邮件提醒, width=30, justify=LEFT)    ###
收件人邮箱输入.grid(row=3, column=1, padx=5, pady=5, sticky=界面.W+E)
收件人邮箱输入.insert(0, 用户配置['邮件设置']['收件人邮箱'][0])

森空岛设置 = 界面.Frame(提醒设置, relief=界面.RIDGE, borderwidth=10)
森空岛设置.grid(row=3, column=0, padx=10, pady=5, sticky=界面.W+E)
森空岛签到开关输入 = 界面.BooleanVar(value=False)    ###
if 用户配置['森空岛签到开关'] == '开': 森空岛签到开关输入.set(True)
界面.Checkbutton(森空岛设置, bootstyle="round-toggle", text="森空岛签到", variable=森空岛签到开关输入, onvalue=TRUE, offvalue=FALSE).grid(row=0, column=0, padx=10, pady=5, sticky=界面.W)
森空岛小秘书开关输入 = 界面.BooleanVar(value=False)    ###
if 用户配置['悬浮字幕开关'] == '开': 森空岛小秘书开关输入.set(True)
界面.Checkbutton(森空岛设置, bootstyle="round-toggle", text="森空岛小秘书", variable=森空岛小秘书开关输入, onvalue=TRUE, offvalue=FALSE).grid(row=1, column=0, padx=10, pady=5, sticky=界面.W)
界面.Label(森空岛设置, text="登录凭证", width=16).grid(row=2, column=0, padx=10, pady=5, sticky=界面.E)
登录凭证输入 = 界面.Entry(森空岛设置, width=30, justify=LEFT, show="●")    ###
登录凭证输入.grid(row=2, column=1, padx=5, pady=5, sticky=界面.W+E)
登录凭证输入.insert(0, 用户配置['登录凭证'])
界面.Label(森空岛设置, text="手机号", width=16).grid(row=3, column=0, padx=10, pady=5, sticky=界面.E)
手机号输入 = 界面.Entry(森空岛设置, width=30, justify=LEFT)    ###
手机号输入.grid(row=3, column=1, padx=5, pady=5, sticky=界面.W+E)
手机号输入.insert(0, 用户配置['手机号'])
界面.Label(森空岛设置, text="密码", width=16).grid(row=4, column=0, padx=10, pady=5, sticky=界面.E)
密码输入 = 界面.Entry(森空岛设置, width=30, justify=LEFT, show="●")    ###
密码输入.grid(row=4, column=1, padx=5, pady=5, sticky=界面.W+E)
密码输入.insert(0, 用户配置['密码'])

# 跑单设置
跑单设置 = 界面.LabelFrame(滚动区域[1], text="跑单设置", relief=界面.RIDGE, borderwidth=10)
跑单设置.grid(row=0, column=1, sticky=界面.W+E+N+S, padx=10, pady=10)
界面.Label(跑单设置, text="跑单提前运行时间").grid(row=0, column=0, padx=10, pady=5, sticky=界面.W)
跑单提前运行时间输入 = 界面.Entry(跑单设置, width=15, justify=LEFT)
跑单提前运行时间输入.grid(row=0, column=1, padx=5, pady=5, columnspan=2, sticky=界面.W+E)
跑单提前运行时间输入.insert(0, 用户配置['跑单提前运行时间'])
界面.Label(跑单设置, text="更换干员前缓冲时间").grid(row=1, column=0, padx=10, pady=5, sticky=界面.W)
更换干员前缓冲时间输入 = 界面.Entry(跑单设置, width=15, justify=LEFT)    ###
更换干员前缓冲时间输入.grid(row=1, column=1, padx=5, pady=5, columnspan=2, sticky=界面.W+E)
更换干员前缓冲时间输入.insert(0, 用户配置['更换干员前缓冲时间'])
跑单消耗无人机开关输入 = 界面.BooleanVar(value=False)    ###
if 用户配置['跑单消耗无人机开关'] == '开': 跑单消耗无人机开关输入.set(True)
界面.Checkbutton(跑单设置, bootstyle="round-toggle", text="无人机辅助跑单", variable=跑单消耗无人机开关输入, onvalue=TRUE, offvalue=FALSE).grid(row=2, column=0, padx=10, pady=5, sticky=界面.W)
任务结束后退出游戏开关输入 = 界面.BooleanVar(value=False)    ###
if 用户配置['任务结束后退出游戏'] == '是': 任务结束后退出游戏开关输入.set(True)
界面.Checkbutton(跑单设置, bootstyle="round-toggle", text="任务结束后退出游戏", variable=任务结束后退出游戏开关输入, onvalue=TRUE, offvalue=FALSE).grid(row=3, column=0, padx=10, pady=5, sticky=界面.W)

# 龙舌兰和但书自动休息设置
龙舌兰和但书自动休息设置 = 界面.LabelFrame(滚动区域[1], text="龙舌兰和但书自动休息设置", relief=界面.RIDGE, borderwidth=10)
龙舌兰和但书自动休息设置.grid(row=1, column=1, sticky=界面.W+E+N+S, padx=10, pady=10)
龙舌兰和但书自动休息开关输入 = 界面.BooleanVar(value=False)    ###
if 用户配置['龙舌兰和但书休息'] == '开': 龙舌兰和但书自动休息开关输入.set(True)
界面.Checkbutton(龙舌兰和但书自动休息设置, bootstyle="round-toggle", text="龙舌兰和但书自动休息", variable=龙舌兰和但书自动休息开关输入, onvalue=TRUE, offvalue=FALSE).grid(row=0, column=0, columnspan=6, padx=10, pady=5, sticky=界面.W)
界面.Label(龙舌兰和但书自动休息设置, text="门牌号").grid(row=1, column=0, padx=10, pady=5, sticky=界面.W)
宿舍门牌号输入 = 界面.Entry(龙舌兰和但书自动休息设置, justify=LEFT, width=15)    ###
宿舍门牌号输入.grid(row=1, column=1, columnspan=5, padx=5, pady=5, sticky=界面.W)
宿舍门牌号输入.insert(0, list(用户配置['宿舍设置'].keys())[0])
界面.Label(龙舌兰和但书自动休息设置, text="干员安排").grid(row=2, column=0, padx=10, pady=5, sticky=界面.W)
宿舍干员安排 = []
for 序号 in range(5):
    宿舍干员安排.append(界面.Entry(龙舌兰和但书自动休息设置, justify=LEFT, width=11))
    宿舍干员安排[序号].grid(row=2, column=序号+1, padx=5, pady=5, sticky=界面.W)
    宿舍干员安排[序号].insert(0, 用户配置['宿舍设置'][list(用户配置['宿舍设置'].keys())[0]][序号])

# MAA作战设置
MAA作战设置 = 界面.LabelFrame(滚动区域[1], text="MAA作战设置", relief=界面.RIDGE, borderwidth=10)
MAA作战设置.grid(row=2, column=1, sticky=界面.W+E+N+S, padx=10, pady=10)
MAA作战开关输入 = 界面.BooleanVar(value=False)    ###
if 用户配置['MAA设置']['作战开关'] == '开': MAA作战开关输入.set(True)
界面.Checkbutton(MAA作战设置, bootstyle="round-toggle", text="MAA作战", variable=MAA作战开关输入, onvalue=TRUE, offvalue=FALSE).grid(row=0, column=0, columnspan=2, padx=10, pady=5, sticky=界面.W)
界面.Label(MAA作战设置, text="MAA路径").grid(row=1, column=0, padx=10, pady=5, sticky=界面.W)
MAA路径输入 = 界面.Entry(MAA作战设置, justify=LEFT, width=40)    ###
MAA路径输入.grid(row=1, column=1, padx=5, pady=5, sticky=界面.W)
MAA路径输入.insert(0, 用户配置['MAA设置']['MAA路径'])
界面.Label(MAA作战设置, text="MAA adb路径").grid(row=2, column=0, padx=10, pady=5, sticky=界面.W)
MAA_adb路径输入 = 界面.Entry(MAA作战设置, justify=LEFT, width=40)    ###
MAA_adb路径输入.grid(row=2, column=1, padx=5, pady=5, sticky=界面.W)
MAA_adb路径输入.insert(0, 用户配置['MAA设置']['MAA_adb路径'])

程序特点域 = 界面.Frame(滚动区域[2], relief=界面.RIDGE, borderwidth=0)
集成战略开关输入 = 界面.BooleanVar(value=False)    ###
if 用户配置['MAA设置']['集成战略'] == '开': 集成战略开关输入.set(True)
界面.Checkbutton(MAA作战设置, bootstyle="round-toggle", text="集成战略", variable=集成战略开关输入, onvalue=TRUE, offvalue=FALSE).grid(row=3, column=0, columnspan=2, padx=10, pady=5, sticky=界面.W)
集成战略主题输入 = 界面.StringVar(value=用户配置['MAA设置']['集成战略主题'])  ###
界面.Radiobutton(MAA作战设置, text='探索者的银凇止境', variable=集成战略主题输入, value='探索者的银凇止境').grid(row=3, column=1, padx=10, pady=5, sticky=界面.W)
界面.Radiobutton(MAA作战设置, text='水月与深蓝之树', variable=集成战略主题输入, value='水月与深蓝之树').grid(row=4, column=1, padx=10, pady=5, sticky=界面.W)
界面.Radiobutton(MAA作战设置, text='傀影与猩红孤钻', variable=集成战略主题输入, value='傀影与猩红孤钻').grid(row=5, column=1, padx=10, pady=5, sticky=界面.W)
界面.Label(MAA作战设置, text="集成战略分队").grid(row=6, column=0, padx=10, pady=5, sticky=界面.W)
集成战略分队输入 = 界面.Entry(MAA作战设置, justify=LEFT, width=15)    ###
集成战略分队输入.grid(row=6, column=1, padx=5, pady=5, sticky=界面.W)
集成战略分队输入.insert(0, 用户配置['MAA设置']['集成战略分队'])
界面.Label(MAA作战设置, text="集成战略开局招募组合").grid(row=7, column=0, padx=10, pady=5, sticky=界面.W)
集成战略开局招募组合输入 = 界面.Entry(MAA作战设置, justify=LEFT, width=15)    ###
集成战略开局招募组合输入.grid(row=7, column=1, padx=5, pady=5, sticky=界面.W)
集成战略开局招募组合输入.insert(0, 用户配置['MAA设置']['集成战略开局招募组合'])
界面.Label(MAA作战设置, text="集成战略开局干员").grid(row=8, column=0, padx=10, pady=5, sticky=界面.W)
集成战略开局干员输入 = 界面.Entry(MAA作战设置, justify=LEFT, width=15)    ###
集成战略开局干员输入.grid(row=8, column=1, padx=5, pady=5, sticky=界面.W)
集成战略开局干员输入.insert(0, 用户配置['MAA设置']['集成战略开局干员'])
界面.Label(MAA作战设置, text="集成战略策略模式").grid(row=9, column=0, padx=10, pady=5, sticky=界面.W)
集成战略策略模式输入 = 界面.StringVar(value=用户配置['MAA设置']['集成战略策略模式'])  ###
界面.Radiobutton(MAA作战设置, text='刷等级', variable=集成战略策略模式输入, value='0').grid(row=9, column=1, padx=10, pady=5, sticky=界面.W)
界面.Radiobutton(MAA作战设置, text='刷源石锭', variable=集成战略策略模式输入, value='1').grid(row=10, column=1, padx=10, pady=5, sticky=界面.W)
# 生息演算开关输入 = 界面.BooleanVar(value=False)    ###
# if 用户配置['MAA设置']['生息演算'] == '开': 生息演算开关输入.set(True)
# 界面.Checkbutton(MAA作战设置, bootstyle="round-toggle", text="生息演算", variable=生息演算开关输入, onvalue=TRUE, offvalue=FALSE).grid(row=15, column=0, columnspan=2, padx=10, pady=5, sticky=界面.W)
界面.Separator(MAA作战设置, bootstyle="light").grid(row=16, column=0, padx=20, pady=10, sticky=界面.W+E)
界面.Label(MAA作战设置, text="消耗理智关卡").grid(row=17, column=0, padx=10, pady=5, sticky=界面.W)
消耗理智关卡输入 = 界面.Entry(MAA作战设置, justify=LEFT, width=15)    ###
消耗理智关卡输入.grid(row=17, column=1, padx=5, pady=5, sticky=界面.W)
消耗理智关卡输入.insert(0, 用户配置['MAA设置']['消耗理智关卡'])
界面.Label(MAA作战设置, text="每次MAA使用理智药数量").grid(row=18, column=0, padx=10, pady=5, sticky=界面.W)
每次MAA使用理智药数量输入 = 界面.Entry(MAA作战设置, justify=LEFT, width=15)    ###
每次MAA使用理智药数量输入.grid(row=18, column=1, padx=5, pady=5, sticky=界面.W)
每次MAA使用理智药数量输入.insert(0, 用户配置['MAA设置']['使用理智药数量'])

# 使用说明
程序特点 = """欢迎博士使用Mower0！
该程序的特点是：
1. 可选跑单模式
    ①无人机辅助跑单，跑单流程更快，搭配跑单后关闭游戏更省电
    ②常态不消耗无人机跑单，仅在两站接单时间接近时使用无人机拉开接单时间差距，更节省无人机
    跑单后可选全自动休息龙舌兰、但书。
2. 不需要排班表！让体验跑单不再以排班表为门槛！
    仅需要指定龙舌兰、但书的跑单位置，对新手更容易上手来体验进阶的跑单功能；
    对极限玩家兼容更高上限的复杂手动操作；
    对MAA过渡玩家也可以兼容使用MAA排班，但可能面临程序间争夺adb的情况，需要交替手动启动和关闭。
3. 支持一系列森空岛相关功能，如自动签到、游戏数据查看、干员阵容消耗经验龙门币分析功能。
    能够输出一个.csv文件记录干员阵容、练度和消耗经验龙门币供更加深入的分析。
4. 谨防集批悲剧（风雪蔽目.jpg）！可选用通知中心的弹窗提示和桌面悬浮任务提示字幕（类似音乐软件的桌面歌词）。
    悬浮字幕可以把鼠标放在字上，通过滚轮轻松调节字幕大小，双击字幕可以关闭，关闭后也可以在托盘图标处再次打开。
    眼看作战没有结束，任务又马上要开始的情况也可以在托盘处暂时先停止Mower0。
    双击Mower0的托盘图标会通过弹窗通知查看当前跑单任务信息。"""
程序特点域 = 界面.Frame(滚动区域[2], relief=界面.RIDGE, borderwidth=0)
程序特点域.grid(row=0, column=0, sticky=界面.W+E+N+S, padx=10, pady=10)
程序特点行列表 = 程序特点.split('\n')
for 行号, 行 in enumerate(程序特点行列表):
    if 行[1] == ".": 界面.Label(程序特点域, text=行, font="微软雅黑 9 bold", foreground="mediumpurple").grid(row=行号, column=0, padx=10, pady=5, sticky=界面.W)
    else: 界面.Label(程序特点域, text=行).grid(row=行号, column=0, padx=10, pady=5, sticky=界面.W)
界面.Separator(滚动区域[2], bootstyle="dark").grid(row=1, column=0, padx=20, pady=5, sticky=界面.W+E)
使用流程 = """Mower0的使用流程：
1. 请检查模拟器的分辨率，Mower0仅支持16:9比例，1600×900及以上分辨率。请设置模拟器的性能保持的帧率不小于30帧，便于进行识别。
2. 按照自身情况仿照给出的范例进行修改，请至少填写主页的设置。
3. 开始运行!
4. 初次使用Mower0的初次点击开始运行后，会进行一些必要工具包的下载，请耐心等待。
※※※ 请注意，在贸易站效率有任何变化后（例如换班）请记得重新运行Mower0!"""
使用流程域 = 界面.Frame(滚动区域[2], relief=界面.RIDGE, borderwidth=0)
使用流程域.grid(row=2, column=0, sticky=界面.W+E+N+S, padx=10, pady=10)
使用流程行列表 = 使用流程.split('\n')
for 行号, 行 in enumerate(使用流程行列表): 界面.Label(使用流程域, text=行, font="微软雅黑 9 bold", foreground="indigo").grid(row=行号, column=0, padx=10, pady=5, sticky=界面.W)


托盘菜单 = (MenuItem(任务提示, 跑单任务查询, default=True, visible=False),
            MenuItem('显示字幕', 显示字幕, visible=悬浮字幕开关),
            Menu.SEPARATOR,
            MenuItem('森空岛签到', 森空岛签到, visible=签到),
            MenuItem('森空岛查看游戏内信息', 森空岛查看游戏内信息, visible=森空岛小秘书),
            MenuItem('森空岛干员阵容查询', 森空岛干员阵容查询, visible=森空岛小秘书),
            Menu.SEPARATOR,
            MenuItem('重新运行Mower0', 重新运行Mower0, visible=True),
            MenuItem('停止运行Mower0', 停止运行Mower0, visible=True),
            Menu.SEPARATOR,
            MenuItem('退出Mower0', 退出Mower0))
托盘图标 = pystray.Icon("Mower0", Image.open("元素/图标.png"), "Mower0", 托盘菜单)


if 悬浮字幕开关:
    字幕窗口 = Toplevel(窗口)
    字幕窗口.withdraw()
    窗口宽度 = 字幕窗口.winfo_screenwidth()
    窗口高度 = 字幕窗口.winfo_screenheight()
    字幕字号 = 窗口高度 // 24
    字幕窗口.geometry("%dx%d+%d+%d" % (窗口宽度, 窗口高度, (字幕窗口.winfo_screenwidth() - 窗口宽度) / 2,
                                   字幕窗口.winfo_screenheight() * 3 / 4 - 窗口高度 / 2))
    字幕窗口.overrideredirect(True)
    字幕窗口.title("字幕窗口")
    字幕窗口.attributes("-topmost", 1)
    字幕窗口.wm_attributes("-transparentcolor", 字幕颜色)

    # 添加一个悬浮字幕控件
    悬浮字幕 = Label(字幕窗口)
    悬浮字幕.pack(side="top", fill="both", expand=True)
    悬浮字幕.bind("<Button-1>", 选中窗口)
    悬浮字幕.bind("<B1-Motion>", 拖动窗口)
    悬浮字幕.bind("<Double-Button-1>", 关闭窗口)
    悬浮字幕.bind("<MouseWheel>", 缩放字幕)

if __name__ == "__main__":
    日志设置()
    init_fhlr(运行信息滚动窗)
    threading.Thread(target=托盘图标.run, daemon=False).start()
    刷新跑单位置设置()
    if 悬浮字幕开关: 更新字幕()
    mainloop()
