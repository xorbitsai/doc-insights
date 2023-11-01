import logging
import re


class Utf8DecoderFormatter(logging.Formatter):
    def format(self, record):
        original_message = super().format(record)

        def decode_match(match):
            escaped_str = match.group(0)
            # 直接转义
            return escaped_str.encode("utf-8").decode("unicode_escape")

        # 使用正则表达式匹配 Unicode 转义序列
        decoded_message = re.sub(r"\\u[0-9a-fA-F]{4}", decode_match, original_message)
        return decoded_message
