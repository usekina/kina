from ask_sdk_core.skill_builder import SkillBuilder

sb = SkillBuilder()

@sb.request_handler(can_handle_func=lambda handler_input: handler_input.request_envelope.request.type == "LaunchRequest")
def launch_request_handler(handler_input):
    return handler_input.response_builder.speak("Hello World!").get_response()

sb.add_request_handler(launch_request_handler)

if __name__ == "__main__":
    sb.add_request_handler(launch_request_handler)
    lambda_handler = sb.lambda_handler()
