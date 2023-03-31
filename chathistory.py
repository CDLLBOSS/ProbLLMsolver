class ChatHistory:
    def __init__(self):
        self.history = []

    def add_input(self, preprocessed_input):
        self.history.append({"role": "user", "content": preprocessed_input})

    def add_output(self, postprocessed_output):
        self.history.append({"role": "assistant", "content": postprocessed_output})

    def get_history(self):
        return self.history
