# RenderDoc Integration

https://renderdoc.org/docs/in_application_api.html

RenderDoc graphics debugger is linked dynamically at program start. The RenderDoc overlay should appear in program. 

Press Home key to launch debugger UI.

API:
    RDOCAPI->LaunchReplayUI(1, "");
    if (RDOCAPI) RDOCAPI->StartFrameCapture(NULL, NULL);
    if (RDOCAPI) RDOCAPI->EndFrameCapture(NULL, NULL);

Captures saved to C:\Users\Kevin\AppData\Local\Temp\RenderDoc but can be changed with RDOCAPI->SetCaptureFilePathTemplate.
