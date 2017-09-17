from __future__ import print_function
from PIL import Image
import sys
import os
import tempfile
import urllib.parse
from bottle import route, run, request
from fbrecog import FBRecog
import logging
import base64


stuff = {}
logging.basicConfig(level=logging.DEBUG)
fb_token = os.environ["FB_TOKEN"]
fb_dtsg = os.environ["FB_DTSG_TOKEN"]
fb_cookie = os.environ["FB_COOKIE"]
assert fb_token is not None
assert fb_dtsg is not None
assert fb_cookie is not None
recog = FBRecog(fb_token, fb_cookie, fb_dtsg)

with open('../data.txt', 'r') as f:
    for line in f.readlines():
        user_stuff = tuple(line.split(','))
        text = user_stuff[1]
        if len(user_stuff) >= 3:
            extras = ''.join(user_stuff[2:])
            firstfewlines = '\n'.join(extras.split('\t')[:3])
            text += '\n' + firstfewlines

        stuff[user_stuff[1]] = text


@route("/", method='POST')
def root():
    data = request.body.read()
    assert data
    data = base64.b64decode(data)
    width = int.from_bytes(data[-8:-4], byteorder='little')
    height = int.from_bytes(data[-4:], byteorder='little')
    fd, yuv_path = tempfile.mkstemp(prefix='nsar', suffix='.yuv')
    os.close(fd)
    with open(yuv_path, "wb") as f:
        f.write(data[:-8])
    bmp_path = yuv_path.split('.')[0] + '.png'
    os.system("convert -size {}x{} -depth 8 {} {}".format(width, height, yuv_path, bmp_path))
    faces = recog.recognize(bmp_path)
    name = faces[0]['name'] if faces else 'Anonymous'
    text = stuff.get(name, str(name))
    print(text)
    os.remove(yuv_path)

    return text


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'run':
        run(host='0.0.0.0', port=9999, debug=True)
    else:
        print('invalid command {}'.format(sys.argv[1]))
