class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        self.is_built = False # برای بررسی اینکه آیا وزن‌ها مقداردهی شده‌اند

    def forward(self, input_data, training=True):
        raise NotImplementedError

    def backward(self, output_gradient):
        raise NotImplementedError
