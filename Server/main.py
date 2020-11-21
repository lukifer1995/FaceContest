# 標準ライブラリ

# 関連外部ライブラリ
import ssl

from flask import Flask, render_template, Response
from flask_uploads import configure_uploads

from api.api import api
from engine.route import engine
# 内部ライブラリ
from uploads.route import files, uploads
import os
# アプリケーションのインスタンス生成
app = Flask(__name__,
            instance_relative_config=True,
            static_folder="../Client/static",
            template_folder="../Client/templates")

# アプリケーションの設定
# sslの設定
# something got error : `file not found`
dir_path = os.path.dirname(os.path.realpath(__file__))
print("dir_path : ", dir_path)
cert_path =   os.path.join(dir_path, 'cert.crt')
secret_path = os.path.join(dir_path, 'server_secret_wo_pass.key')
config_path = os.path.join(dir_path, 'instance/flask.cfg')

context = ssl.SSLContext(ssl.PROTOCOL_TLSv1_2)
context.load_cert_chain(cert_path, secret_path)
app.config.from_pyfile(config_path)

# ファイルをアップロードできるための設定
configure_uploads(app, files)

# ルートの設定
app.register_blueprint(uploads)
app.register_blueprint(engine)
app.register_blueprint(api)


ssl._create_default_https_context = ssl._create_unverified_context

# URLの設定
@app.route('/')
def main():
    return render_template('index.html')


if __name__ == "__main__":
    # アプリケーション開始
    # TODO: Trungとの確認が必要
    # 確認内容: threaded=Trueに変えた方が良い
    app.run(host='127.0.0.1', port=3000, ssl_context=context, threaded=False, debug=False)



    # Làm sao để transfer learning lưu dataset lại để train tiếp khi có dataset mới
    # File weight.h5 có phình to ra sau nhiều lần train k ?