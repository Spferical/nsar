from __future__ import print_function
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
    print(data)
    fd, path = tempfile.mkstemp(prefix='nsar')
    os.close(fd)
    with open(path, "w") as f:
        f.write(str(data))
    print(recog.recognize(path))
    os.remove(path)

    #TODO: everything
    name = 'Anonymous'

    return {name: name}


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'run':
        run(host='0.0.0.0', port=9999, debug=True)
    else:
        print('invalid command {}'.format(sys.argv[1]))
