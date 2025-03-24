### Решение задачи n тел на GPU

В данном репозитории находится сервер и клиент для решения задачи n [тел](https://ru.wikipedia.org/wiki/%D0%93%D1%80%D0%B0%D0%B2%D0%B8%D1%82%D0%B0%D1%86%D0%B8%D0%BE%D0%BD%D0%BD%D0%B0%D1%8F_%D0%B7%D0%B0%D0%B4%D0%B0%D1%87%D0%B0_N_%D1%82%D0%B5%D0%BB) 

Сервер представляет собой приложение, написанное на C++ использованием технологии [CUDA](https://ru.wikipedia.org/wiki/CUDA) и [grpc](https://grpc.io/)

Клиент представляет собой питон приложение, которому на вход можно подать число тел, константу интегрирование и количество итераций.

Подробнее каждый блок описан в соответствующем подразделе

#### Server

Сервер упакован в docker [контейнер](https://www.docker.com/). Чтобы сервер работал необходимо также скачать [nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

Для сборки и запуска сервера переходим в директорию `server`

Выполняем следующее команды

```bash
docker build . --tag=<your_image_tag>
docker run -it --gpus=all -p <any_port>:9999 <your_image_tag>
```

#### Client

Для запуска клиента переходим в директорию `client`. Далее создаем виртуальное окружение и качаем зависимости из файла `requirements.txt` с помощью команды

```bash
pip install -r requirements.txt
```

После этого необходимо сгенерировать protobuf и grpc зависимости клиента. Для этого используем скрипт `generate_grpc_client.sh`

```
./generate_grpc_client.sh
```

После этого можем запускать клиента. Для этого используем команду

```
python3 client.py --num-bodies <число_тел> --dt <шаг_интегрирования> --num-iterations <число_итераций>
```

На выходе получим файл с анимацией решения задачи. Пример такого файла лежит по пути `animations/n_body_problem_animation.gif`