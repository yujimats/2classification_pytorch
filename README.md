# はじめに
`Pytorch`による画像の2クラス分類AI。  
データは下記のようにダウンロードし、任意の保存先に保存する。  
```bash
wget https://thor.robots.ox.ac.uk/~vgg/data/pets/images.tar.gz -P data/pets
tar -xzf data/pets/images.tar.gz -C data/pets
rm data/pets/images.tar.gz
```

# 動作環境
- MacOS Ventura 13.4.1  
- DockerDesktop 4.21.1  

※上記環境以外の動作は未確認のためご了承ください。  

# 動作方法
## Dockerコンテナの準備
任意のDockerコンテナを立ち上げる。  
必要なライブラリ等は[docker/README.md](docker/README.md)参照のこと。  
データセットとリンクさせるため、Dockerコンテナを立ち上げる際に以下のオプションを付け加える。  
`/path/to/dataset/`にはダウンロードしたデータの保存先を指定する。  
```bash
--volume /path/to/dataset/:/home/2classification_pytorch/:ro
```
※ `/data/pets/images/`をリンクさせるようにすれば動きます。  
## プログラムの実行
Dockerコンテナ内で以下を実行する。  
```bash
sh run.sh
```
学習が開始され、結果が`output`に保存される。  

# 出力結果
学習時の`train`と`validation`のログ、並びに`inference`の結果が保存される。  

# ライセンス
当ライセンスは [MIT ライセンス](./LICENSE)の規約に基づいて付与されています
