# Demo
## Running it
To run the demo, run the following commands:
### setup.sh
```bash 
./setup.sh
```

This sets up the virtual environment along with the necessary stuff for the nlp part.

### Running the demo
```bash
python3 demo.py
```
This is to run the script.
- To see what the camera is seeing, go to `temp/input`
- To see the output of the YOLO model, go to `temp/output`
- To talk with the llm, type in the terminal

## Deleting it
To delete all the installed models, version of llama and virtual environment, run the following command:
```bash
./remove.sh
```
This removes everything installed in `setup.sh`.