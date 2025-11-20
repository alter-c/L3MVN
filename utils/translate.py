import os
import json
import random
import requests
from hashlib import md5


class ObjectNavTranslate:
    """Translate from English to Chinese with Baidu API."""
    def __init__(self,
                 cache_path: str = "datasets/translate.json",
                 from_lang: str = "en",
                 to_lang: str = "zh"):

        self.appid = os.getenv("BAIDU_APPID")
        self.appkey = os.getenv("BAIDU_APPKEY")
        self.from_lang = from_lang
        self.to_lang = to_lang
        self.cache_path = cache_path
        self._load_cache()

        # Baidu API 
        self.endpoint = "http://api.fanyi.baidu.com"
        self.path = "/api/trans/vip/translate"
        self.url = self.endpoint + self.path

    @staticmethod
    def _md5(s: str):
        return md5(s.encode("utf-8")).hexdigest()
    
    def _load_cache(self):
        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self.cache = json.load(f)
        else:
            self.cache = {}

    def _update_cache(self):
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(self.cache, f, ensure_ascii=False, indent=2)

    def translate(self, text: str) -> str:
        """Translate object text with cache."""
        if text in self.cache:
            return self.cache[text]

        salt = random.randint(32768, 65536)
        sign = self._md5(self.appid + text + str(salt) + self.appkey)
        payload = {
            "appid": self.appid,
            "q": text,
            "from": self.from_lang,
            "to": self.to_lang,
            "salt": salt,
            "sign": sign,
        }

        # Translate
        try:
            response = requests.post(self.url, params=payload, timeout=6)
            data = response.json()
            chinese = data["trans_result"][0]["dst"]
        except Exception as e:
            print(f"[WARN] Translation failed: {e}")
            chinese = text  # Fallback to original text

        # Write to cache, avoid re-translation
        self.cache[text] = chinese
        self._update_cache()

        return chinese
    

if __name__ == "__main__":
    trans = ObjectNavTranslate(cache_path="test.json")
    res = trans.translate("sofa")
    print(res)
