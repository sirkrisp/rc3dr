
# Imports
import socketio
import tornado.ioloop
import tornado.web
import os

import argparse
from rc3dr.rc3dr import RC3DR
import numpy as np

parser = argparse.ArgumentParser(description='Start a dash server.')
parser.add_argument('--port', type=int, dest="port",
                    help='The port on which the server runs.')

sio = socketio.AsyncServer(async_mode='tornado') # Important to define async_mode = 'tornado' instead of 'aiohttp'

@sio.event
async def broadcast(sid, msg):
    print("forward")
    print(msg)
    print(msg['event'])
    print(type(msg))
    await sio.emit(msg['event'], msg['data'])

@sio.event
async def client_connected(sid, msg):
    print(msg["data"])

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render("dash.html", title="Dash")

class Dash(object):

    def __init__(self, port):
        super().__init__()

        self.sio = sio
        self.port = port

        # robot arm
        rev_config1 = { "Link" : [0,0,1.45], "RotAxis" : [0,0,1], "AngleLimits" : [0, 2*np.pi]}
        rev_config2 = { "Link" : [0,0,1.45], "RotAxis" : [0,1,0], "AngleLimits" : [0, 2*np.pi]}
        rev_config3 = { "Link" : [0,0,2.7], "RotAxis" : [0,1,0], "AngleLimits" : [0, 2*np.pi]}
        rev_config4 = { "Link" : [1.35,0,0], "RotAxis" : [1,0,0], "AngleLimits" : [0, 2*np.pi]}
        rev_config5 = { "Link" : [1.35,0,0], "RotAxis" : [0,1,0], "AngleLimits" : [0, 2*np.pi]}
        rev_config6 = { "Link" : [0.7,0,0], "RotAxis" : [1,0,1], "AngleLimits" : [0, 2*np.pi]}
        robot_arm_config = [("Revolute", rev_config1),
                            ("Revolute", rev_config2),
                            ("Revolute", rev_config3),
                            ("Revolute", rev_config4),
                            ("Revolute", rev_config5),
                            ("Revolute", rev_config6)]
        # self.robot_arm_handle = RobotArmHandler('/robot-control', robot_arm_config)
        # self.sio.register_namespace(self.robot_arm_handle)

        # self.robot_depth_image_handle = RobotDepthImageHandler('/depth-camera')
        # self.sio.register_namespace(self.robot_depth_image_handle)

        self.rc3dr = RC3DR('/rc3dr')
        self.sio.register_namespace(self.rc3dr)

        settings = {
            "static_path": os.path.join(os.path.dirname(__file__), "static"),
            "template_path" : os.path.join(os.path.dirname(__file__), "templates"),
        }

        app = tornado.web.Application([
            (r"/", MainHandler),
            (r"/socket.io/", socketio.get_tornado_handler(sio)),
            (r'/media/models/obj/(.*)', tornado.web.StaticFileHandler, {'path': './media/models/obj'}),
            (r'/media/models/urdf/(.*)', tornado.web.StaticFileHandler, {'path': './media/models/urdf'}),

        ], **settings) # passing static_path="..." automatically creates tornado.web.StaticFileHandler

        
        app.listen(port)
        
    
    def start_server(self):
        print("======== Running on http://0.0.0.0:%i ========" % self.port)
        print("Press CTRL+C to quit")
        tornado.ioloop.IOLoop.current().start()

    @sio.event
    async def broadcast(self, msg):
        print("forward")
        print(msg)
        print(msg['event'])
        print(type(msg))
        await sio.emit(msg['event'], msg['data'])
                
# Virtual Card Class

class Card(object):

    def __init__(self):
        super().__init__()

if __name__ == '__main__':
    args = parser.parse_args()
    dash = Dash(**vars(args))
    dash.start_server()
    
