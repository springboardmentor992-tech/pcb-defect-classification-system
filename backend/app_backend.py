from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from Inspection_ReNet import run_inspection
import base64
import cv2

# ✅ CREATE APP FIRST
app = FastAPI(title="PCB Inspection API")


@app.post("/inspect")
async def inspect_pcb(
    template: UploadFile = File(...),
    test: UploadFile = File(...)
):

    template_path = "temp_template.jpg"
    test_path = "temp_test.jpg"

    try:
        # Save uploaded images
        with open(template_path, "wb") as f:
            f.write(await template.read())

        with open(test_path, "wb") as f:
            f.write(await test.read())

        # Run inspection
        detections, aligned_image = run_inspection(template_path, test_path)

        # Convert aligned image to base64
        _, buffer = cv2.imencode(".jpg", aligned_image)
        image_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "status": "success",
            "detections": detections,
            "aligned_image": image_base64
        }

    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        })
