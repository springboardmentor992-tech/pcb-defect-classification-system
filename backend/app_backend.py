import os
from fastapi import FastAPI, UploadFile, File
from Inspection_ReNet import run_inspection

app = FastAPI(title="PCB Inspection API")

@app.post("/inspect")
async def inspect_pcb(template: UploadFile = File(...), test: UploadFile = File(...)):

    template_path = "temp_template.jpg"
    test_path = "temp_test.jpg"

    with open(template_path, "wb") as f:
        f.write(await template.read())

    with open(test_path, "wb") as f:
        f.write(await test.read())

    detections = run_inspection(template_path, test_path)

    return {
        "status": "success",
        "detections": detections
    }