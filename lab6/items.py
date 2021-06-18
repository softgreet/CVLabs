import json
'''
信息封装类
'''
class MessageItem(object):
 #用于封装信息的类,包含图片和其他信息
 def __init__(self,frame,message):
  self._frame = frame
  self._message = message
 def getFrame(self):
  #图片信息
  return self._frame
 def getMessage(self):
  #文字信息,json格式
  return self._message

 def getJson(self):
  #获得json数据格式
  dicdata = {"frame":self.getBase64Frame().decode(),"message":self.getMessage()}
  return json.dumps(dicdata)