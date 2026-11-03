"""Script to generate rich demo clusters for ClutterIQ:
1. Portrait clusters of the same person from different angles, sharpness levels, and resolutions.
2. LeetCode profile screenshots, submissions logs, and contest rating charts.
"""

from pathlib import Path
import shutil
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

DEMO_DIR = Path("demo_showcase")
DEMO_DIR.mkdir(parents=True, exist_ok=True)

BRAIN_DIR = Path(r"C:\Users\aksha\.gemini\antigravity\brain\803033c1-567a-4c33-8137-5c34fffe54a0")

# ---------------------------------------------------------------------------
# 1. Process Portrait Photos (Different angles, sharpness, and resolution)
# ---------------------------------------------------------------------------
p1_src = BRAIN_DIR / "portrait_1_front_1793715725301.jpg"
p2_src = BRAIN_DIR / "portrait_2_three_quarter_1793715748848.jpg"
p3_src = BRAIN_DIR / "portrait_3_outdoor_daylight_1793715775839.jpg"

if p1_src.exists():
    # A. Front-facing Sharp HD (The Keeper)
    hd_front_path = DEMO_DIR / "Portrait_Elena_Front_Sharp_HD.jpg"
    shutil.copy(p1_src, hd_front_path)
    print(f"Copied Sharp HD portrait -> {hd_front_path.name}")

    # B. Front-facing Blurry (Motion / Out of focus)
    # Using Gaussian blur to drop Laplacian variance below BLUR_THRESHOLD (100.0)
    img_pil = Image.open(p1_src)
    blurry_img = img_pil.filter(ImageFilter.GaussianBlur(radius=5.5))
    blurry_path = DEMO_DIR / "Portrait_Elena_Front_Motion_Blur.jpg"
    blurry_img.save(blurry_path, quality=85)
    print(f"Generated Blurry portrait -> {blurry_path.name}")

    # C. Front-facing Low-Res Compressed
    lowres_img = img_pil.resize((320, 320), Image.Resampling.BILINEAR)
    lowres_path = DEMO_DIR / "Portrait_Elena_Front_LowRes_Thumbnail.jpg"
    lowres_img.save(lowres_path, quality=40)
    print(f"Generated Low-Res portrait -> {lowres_path.name}")

if p2_src.exists():
    # Different angle: Three-quarter smile
    p2_dest = DEMO_DIR / "Portrait_Elena_Angle_3_4_Smile.jpg"
    shutil.copy(p2_src, p2_dest)
    print(f"Copied 3/4 angle portrait -> {p2_dest.name}")

if p3_src.exists():
    # Different angle / environment: Outdoor daylight tilt
    p3_dest = DEMO_DIR / "Portrait_Elena_Outdoor_Daylight.jpg"
    shutil.copy(p3_src, p3_dest)
    print(f"Copied outdoor portrait -> {p3_dest.name}")


# ---------------------------------------------------------------------------
# 2. Generate LeetCode Screenshots & Clusters
# ---------------------------------------------------------------------------
def draw_rounded_rect(draw, bounds, radius, fill, outline=None, width=1):
    x0, y0, x1, y1 = bounds
    draw.rounded_rectangle([x0, y0, x1, y1], radius=radius, fill=fill, outline=outline, width=width)

def create_leetcode_profile_hd() -> Image.Image:
    """Generate high-fidelity dark-mode LeetCode profile screenshot."""
    w, h = 1200, 750
    img = Image.new("RGB", (w, h), color=(26, 26, 26))
    draw = ImageDraw.Draw(img)

    # Navbar
    draw.rectangle([0, 0, w, 55], fill=(38, 38, 38))
    # LeetCode Logo emblem
    draw_rounded_rect(draw, [25, 12, 55, 42], radius=6, fill=(255, 161, 22))
    draw.text((65, 16), "LeetCode", fill=(240, 240, 240))
    draw.text((160, 18), "Explore", fill=(160, 160, 160))
    draw.text((230, 18), "Problems", fill=(255, 255, 255))
    draw.text((315, 18), "Contest", fill=(160, 160, 160))
    draw.text((390, 18), "Discuss", fill=(160, 160, 160))

    # User pill on right
    draw_rounded_rect(draw, [w - 180, 12, w - 25, 42], radius=15, fill=(48, 48, 48))
    draw.ellipse([w - 175, 16, w - 151, 38], fill=(13, 148, 136))
    draw.text((w - 145, 18), "akshath_dev", fill=(255, 255, 255))

    # Left Profile Card
    draw_rounded_rect(draw, [25, 75, 340, 420], radius=12, fill=(34, 34, 34), outline=(50, 50, 50))
    # User Avatar
    draw.ellipse([50, 95, 130, 175], fill=(79, 70, 229))
    draw.text((78, 122), "AD", fill=(255, 255, 255))
    draw.text((145, 105), "Akshath | SWE", fill=(255, 255, 255))
    draw.text((145, 130), "@akshath_dev", fill=(140, 140, 140))
    draw.text((145, 152), "Rank: 42,109", fill=(255, 161, 22))

    # Bio & Stats
    draw.line([45, 195, 320, 195], fill=(50, 50, 50), width=1)
    draw.text((45, 210), "📍 Bengaluru, India", fill=(170, 170, 170))
    draw.text((45, 235), "💼 Core Computer Science & Algorithms", fill=(170, 170, 170))
    draw.text((45, 260), "🔗 github.com/akshath", fill=(59, 130, 246))

    # Badges
    draw.text((45, 295), "Badges (3)", fill=(220, 220, 220))
    draw_rounded_rect(draw, [45, 320, 125, 355], radius=6, fill=(55, 48, 163))
    draw.text((55, 330), "Knight", fill=(244, 244, 245))
    draw_rounded_rect(draw, [135, 320, 225, 355], radius=6, fill=(180, 83, 9))
    draw.text((145, 330), "50 Days '24", fill=(254, 243, 199))
    draw_rounded_rect(draw, [235, 320, 315, 355], radius=6, fill=(21, 128, 61))
    draw.text((245, 330), "100 Days", fill=(220, 252, 231))

    # Right Main Panel: Solved Problems Ring Card
    draw_rounded_rect(draw, [360, 75, 780, 420], radius=12, fill=(34, 34, 34), outline=(50, 50, 50))
    draw.text((385, 95), "Solved Problems", fill=(230, 230, 230))

    # Outer circle ring
    cx, cy, cr = 480, 230, 90
    draw.ellipse([cx - cr, cy - cr, cx + cr, cy + cr], fill=(42, 42, 42), outline=(60, 60, 60), width=3)
    # Circle progress slice (Green easy)
    draw.pieslice([cx - cr, cy - cr, cx + cr, cy + cr], start=0, end=140, fill=(0, 184, 163))
    # Circle progress slice (Yellow medium)
    draw.pieslice([cx - cr, cy - cr, cx + cr, cy + cr], start=140, end=300, fill=(255, 192, 30))
    # Circle progress slice (Red hard)
    draw.pieslice([cx - cr, cy - cr, cx + cr, cy + cr], start=300, end=340, fill=(255, 55, 95))
    # Center cutout for donut
    draw.ellipse([cx - 60, cy - 60, cx + 60, cy + 60], fill=(34, 34, 34))
    draw.text((cx - 28, cy - 20), "482", fill=(255, 255, 255))
    draw.text((cx - 22, cy + 5), "/ 3,350", fill=(130, 130, 130))

    # Stats Breakdown
    draw_rounded_rect(draw, [600, 135, 755, 185], radius=8, fill=(42, 42, 42))
    draw.text((615, 142), "Easy", fill=(0, 184, 163))
    draw.text((615, 162), "198 / 840", fill=(240, 240, 240))

    draw_rounded_rect(draw, [600, 205, 755, 255], radius=8, fill=(42, 42, 42))
    draw.text((615, 212), "Medium", fill=(255, 192, 30))
    draw.text((615, 232), "236 / 1,750", fill=(240, 240, 240))

    draw_rounded_rect(draw, [600, 275, 755, 325], radius=8, fill=(42, 42, 42))
    draw.text((615, 282), "Hard", fill=(255, 55, 95))
    draw.text((615, 302), "48 / 760", fill=(240, 240, 240))

    # Contest Rating Card (Top Right)
    draw_rounded_rect(draw, [800, 75, w - 25, 420], radius=12, fill=(34, 34, 34), outline=(50, 50, 50))
    draw.text((820, 95), "Contest Rating", fill=(230, 230, 230))
    draw.text((820, 130), "1,845", fill=(255, 255, 255))
    draw.text((820, 165), "Global Ranking: 14,208 / 620,000 (Top 7.2%)", fill=(160, 160, 160))
    draw.text((820, 190), "Attended: 24 Contests", fill=(160, 160, 160))

    # Mini sparkline chart for contest rating
    points = [(820, 360), (850, 340), (880, 345), (920, 310), (960, 290), (1000, 300), (1050, 260), (1100, 240), (1140, 225)]
    for i in range(len(points) - 1):
        draw.line([points[i], points[i+1]], fill=(59, 130, 246), width=3)
        draw.ellipse([points[i][0]-3, points[i][1]-3, points[i][0]+3, points[i][1]+3], fill=(96, 165, 250))
    draw.ellipse([points[-1][0]-4, points[-1][1]-4, points[-1][0]+4, points[-1][1]+4], fill=(255, 161, 22))

    # Bottom Submission Heatmap Strip
    draw_rounded_rect(draw, [25, 450, w - 25, 710], radius=12, fill=(34, 34, 34), outline=(50, 50, 50))
    draw.text((45, 470), "514 Submissions in the past year | Current Streak: 42 Days", fill=(220, 220, 220))
    # Grid of calendar squares
    start_x, start_y = 45, 510
    np.random.seed(42)
    colors = [
        (46, 46, 46),
        (13, 68, 41),
        (0, 109, 50),
        (38, 166, 65),
        (57, 211, 83),
    ]
    for col in range(50):
        for row in range(7):
            intensity = np.random.choice([0, 1, 2, 3, 4], p=[0.25, 0.25, 0.25, 0.15, 0.10])
            c = colors[intensity]
            x = start_x + col * 22
            y = start_y + row * 22
            draw_rounded_rect(draw, [x, y, x + 18, y + 18], radius=3, fill=c)

    return img


def create_leetcode_submissions_log() -> Image.Image:
    """Generate screenshot of recent accepted LeetCode submissions table."""
    w, h = 1200, 650
    img = Image.new("RGB", (w, h), color=(26, 26, 26))
    draw = ImageDraw.Draw(img)

    # Navbar
    draw.rectangle([0, 0, w, 55], fill=(38, 38, 38))
    draw_rounded_rect(draw, [25, 12, 55, 42], radius=6, fill=(255, 161, 22))
    draw.text((65, 16), "LeetCode", fill=(240, 240, 240))
    draw.text((160, 18), "Submissions Archive", fill=(255, 255, 255))

    # Main Card
    draw_rounded_rect(draw, [25, 80, w - 25, 610], radius=12, fill=(34, 34, 34), outline=(50, 50, 50))
    draw.text((50, 105), "Recent AC Submissions — akshath_dev", fill=(255, 255, 255))

    # Table Header
    draw.rectangle([50, 145, w - 50, 185], fill=(44, 44, 44))
    draw.text((65, 155), "Status", fill=(180, 180, 180))
    draw.text((220, 155), "Problem Title", fill=(180, 180, 180))
    draw.text((600, 155), "Difficulty", fill=(180, 180, 180))
    draw.text((750, 155), "Runtime", fill=(180, 180, 180))
    draw.text((880, 155), "Memory", fill=(180, 180, 180))
    draw.text((1020, 155), "Language", fill=(180, 180, 180))

    submissions = [
        ("Accepted", "42. Trapping Rain Water", "Hard", "14 ms", "18.2 MB", "Python3", (0, 184, 163), (255, 55, 95)),
        ("Accepted", "146. LRU Cache", "Medium", "48 ms", "74.8 MB", "Python3", (0, 184, 163), (255, 192, 30)),
        ("Accepted", "207. Course Schedule", "Medium", "22 ms", "19.5 MB", "Python3", (0, 184, 163), (255, 192, 30)),
        ("Accepted", "76. Minimum Window Substring", "Hard", "38 ms", "17.9 MB", "Python3", (0, 184, 163), (255, 55, 95)),
        ("Accepted", "1. Two Sum", "Easy", "3 ms", "17.6 MB", "Python3", (0, 184, 163), (0, 184, 163)),
        ("Accepted", "200. Number of Islands", "Medium", "19 ms", "21.4 MB", "Python3", (0, 184, 163), (255, 192, 30)),
        ("Accepted", "23. Merge k Sorted Lists", "Hard", "28 ms", "20.1 MB", "Python3", (0, 184, 163), (255, 55, 95)),
    ]

    for idx, (stat, title, diff, rt, mem, lang, s_color, d_color) in enumerate(submissions):
        y = 200 + idx * 55
        draw.line([50, y - 10, w - 50, y - 10], fill=(45, 45, 45), width=1)
        draw.text((65, y), stat, fill=s_color)
        draw.text((220, y), title, fill=(240, 240, 240))
        draw.text((600, y), diff, fill=d_color)
        draw.text((750, y), rt, fill=(190, 190, 190))
        draw.text((880, y), mem, fill=(190, 190, 190))
        draw.text((1020, y), lang, fill=(140, 140, 140))

    return img


# Create and save LeetCode assets
lc_hd = create_leetcode_profile_hd()
lc_hd_path = DEMO_DIR / "LeetCode_Profile_Akshath_Dark_HD.png"
lc_hd.save(lc_hd_path)
print(f"Generated LeetCode HD profile -> {lc_hd_path.name}")

# Near-duplicate compressed LeetCode profile (slightly blurry / low quality)
lc_compressed = lc_hd.resize((840, 525), Image.Resampling.BILINEAR).filter(ImageFilter.GaussianBlur(radius=1.8))
lc_comp_path = DEMO_DIR / "LeetCode_Profile_Akshath_Compressed.jpg"
lc_compressed.save(lc_comp_path, quality=45)
print(f"Generated LeetCode Compressed near-duplicate -> {lc_comp_path.name}")

# LeetCode crop: Solved stats circle crop
lc_crop = lc_hd.crop((360, 75, 780, 420))
lc_crop_path = DEMO_DIR / "LeetCode_Profile_Solved_Stats_Crop.png"
lc_crop.save(lc_crop_path)
print(f"Generated LeetCode Solved stats crop -> {lc_crop_path.name}")

# LeetCode submissions log
lc_sub = create_leetcode_submissions_log()
lc_sub_path = DEMO_DIR / "LeetCode_Recent_Submissions_Log.png"
lc_sub.save(lc_sub_path)
print(f"Generated LeetCode Submissions log -> {lc_sub_path.name}")

print("All demo showcase cluster files generated successfully!")

