import requests, pathlib, urllib

class ColabInfo:
    def version(self):
        return requests.get('http://172.28.0.2:9000/api').json()['version']

    def _get_info(self):
        return requests.get('http://172.28.0.2:9000/api/sessions').json()[0]

    def notebook_name(self, extension=True):
        raw_name = self._get_info()['name']
        parsed_name = urllib.parse.unquote(raw_name)
        return parsed_name if extension else str(pathlib.Path(parsed_name).with_suffix(''))

    def file_id(self, full_path=False):
        id = self._get_info()['path'].split("=")[1]
        return f"colab.research.google.com/drive/{id}" if full_path else id