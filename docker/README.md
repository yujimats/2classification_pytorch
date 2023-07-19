# はじめに
任意のDockerfileを入れる。  
必要なPythonライブラリは以下  
```
torch==1.12.1
torchvision==0.13.1
```
# Dockerコンテナの準備
任意のDockerコンテナを立ち上げる。  
必要なライブラリ等は[docker/README.md](docker/README.md)参照のこと。  
データセットとリンクさせるため、Dockerコンテナを立ち上げる際に以下のオプションを付け加える。  
`/path/to/dataset/`にはダウンロードしたデータの保存先を指定する。  
```bash
--volume /path/to/dataset/:/home/2classification_pytorch/:ro
```

