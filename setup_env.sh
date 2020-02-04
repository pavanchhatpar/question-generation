python3 -m venv venv
source ./venv/bin/activate && pip install -r requirements.txt
[ -f jupyter.password ] || echo "p@bzyK8YFs5?!Ybe" > jupyter.password
./build_images.sh