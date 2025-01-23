class Response:
    def __init__(self, status, body):
        self.success = status
        self.body = body


def success(body=''):
    return Response(status=True, body=body)


def error(body=''):
    return Response(status=False, body=body)


def bool_response(boolean):
    return success('true') if boolean is True else success('false')


