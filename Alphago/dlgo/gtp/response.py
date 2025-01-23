class Response:
    def __init__(self, status, body):
        self.success = status
        self.body = body


def success(body=''):
    return Response(status=True, body=body)


