def assert_message(filename=None, texts=None):
    if filename is None:
        filename = "Unknown file"
    if texts is None:
        texts = "Undefined error"
    return f"Asserts in Quantizable Transformer project\nAssertion File: [{filename}]\nAssertion Reason: {texts}"