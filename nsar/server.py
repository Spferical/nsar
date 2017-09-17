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

logging.basicConfig(level=logging.DEBUG)
fb_token = os.environ["FB_TOKEN"]
fb_dtsg = os.environ["FB_DTSG_TOKEN"]
fb_cookie = os.environ["FB_COOKIE"]
assert fb_token is not None
assert fb_dtsg is not None
assert fb_cookie is not None
recog = FBRecog(fb_token, fb_cookie, fb_dtsg)


@route("/", method='POST')
def root():
    data = request.body.read()
    print(data)
    assert data
    data = base64.b64decode(data)
    width = int.from_bytes(data[0:4], byteorder='little')
    height = int.from_bytes(data[4:8], byteorder='little')
    img = Image.frombytes('RGB', (width, height), data[8:])
    print(data)
    fd, path = tempfile.mkstemp(prefix='nsar', suffix='.png')
    os.close(fd)
    img.save(path)
    faces = recog.recognize(path)
    os.remove(path)
    name = faces[0]['name'] if faces else 'Anonymous'

    return {name: name}


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'run':
        run(host='0.0.0.0', port=9999, debug=True)
    else:
        print('invalid command {}'.format(sys.argv[1]))
