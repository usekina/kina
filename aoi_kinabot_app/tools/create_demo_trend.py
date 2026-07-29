"""Create the public, clearly labeled KinaBot demo trend visual."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path(__file__).resolve().parents[1] / "assets" / "kinabot-demo-trends-v1.png"
FONT_DIR = Path("C:/Windows/Fonts")

SESSIONS = [1, 2, 3, 4, 5, 6]
SERIES = {
    "Vocabulary Variety": [64, 67, 65, 70, 69, 72],
    "Response Length": [58, 62, 61, 65, 63, 66],
    "Speech Structure": [70, 69, 72, 71, 74, 73],
    "Pause Pattern": [61, 59, 63, 62, 65, 64],
}
COLORS = ["#67E8F9", "#A78BFA", "#60A5FA", "#2DD4BF"]


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "segoeuib.ttf" if bold else "segoeui.ttf"
    return ImageFont.truetype(str(FONT_DIR / name), size)


def main() -> None:
    width, height = 1600, 900
    image = Image.new("RGB", (width, height), "#070B1D")
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((70, 60, 1530, 825), radius=36, fill="#0D1533")
    draw.text((125, 105), "Personal Pattern Trends", fill="#F8FAFC", font=font(46, True))
    draw.text(
        (125, 175),
        "DEMO DATA  •  Six sample sessions",
        fill="#67E8F9",
        font=font(22, True),
    )

    plot_left, plot_top, plot_right, plot_bottom = 160, 310, 1470, 690
    for score in [40, 50, 60, 70, 80]:
        y = plot_bottom - (score - 40) / 45 * (plot_bottom - plot_top)
        draw.line((plot_left, y, plot_right, y), fill="#263451", width=2)
        draw.text((100, y - 13), str(score), fill="#94A3B8", font=font(20))

    x_step = (plot_right - plot_left) / (len(SESSIONS) - 1)
    for index, session in enumerate(SESSIONS):
        x = plot_left + index * x_step
        draw.text(
            (x - 42, plot_bottom + 26),
            f"Session {session}",
            fill="#CBD5E1",
            font=font(19),
        )

    for (label, values), color in zip(SERIES.items(), COLORS):
        points = []
        for index, value in enumerate(values):
            x = plot_left + index * x_step
            y = plot_bottom - (value - 40) / 45 * (plot_bottom - plot_top)
            points.append((x, y))
        draw.line(points, fill=color, width=6, joint="curve")
        for x, y in points:
            draw.ellipse((x - 8, y - 8, x + 8, y + 8), fill=color, outline="#0D1533", width=3)

    legend_positions = [(125, 230), (465, 230), (765, 230), (1085, 230)]
    for (label, _), color, (x, y) in zip(SERIES.items(), COLORS, legend_positions):
        draw.rounded_rectangle((x, y + 5, x + 38, y + 17), radius=6, fill=color)
        draw.text((x + 52, y - 2), label, fill="#E2E8F0", font=font(20))

    draw.text(
        (125, 765),
        "Scores describe submitted speech samples and personal patterns only. "
        "KinaBot is not a medical assessment.",
        fill="#94A3B8",
        font=font(20),
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT, optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()
