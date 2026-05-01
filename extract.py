from PIL import Image
im = Image.open("/Users/kanishkk/.gemini/antigravity/brain/7a0a843a-02f1-434a-bf64-9760b6d49e4b/test_tackle_video_1777614525966.webp")
im.seek(im.n_frames - 1)
im.save("assets/demo_ui.png")
