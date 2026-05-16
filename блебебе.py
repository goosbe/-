import cv2
import numpy as np

class AdvancedColorDetector:
    """Улучшенный детектор доминирующих цветов с K-means и точной классификацией по HSV"""

    def __init__(self, n_clusters=8):
        self.n_clusters = n_clusters
        # Палитра (RGB)
        self.color_palette = {
            'Black': (0, 0, 0),
            'Dark Gray': (64, 64, 64),
            'Gray': (128, 128, 128),
            'Light Gray': (192, 192, 192),
            'White': (255, 255, 255),
            'Dark Red': (139, 0, 0),
            'Red': (255, 0, 0),
            'Deep Pink': (255, 20, 147),
            'Pink': (255, 192, 203),
            'Light Pink': (255, 182, 193),
            'Dark Orange': (255, 140, 0),
            'Orange': (255, 165, 0),
            'Gold': (255, 215, 0),
            'Yellow': (255, 255, 0),
            'Light Yellow': (255, 255, 224),
            'Dark Green': (0, 100, 0),
            'Green': (0, 255, 0),
            'Lime': (0, 255, 127),
            'Light Green': (144, 238, 144),
            'Cyan': (0, 255, 255),
            'Dark Blue': (0, 0, 139),
            'Blue': (0, 0, 255),
            'Light Blue': (173, 216, 230),
            'Purple': (128, 0, 128),
            'Magenta': (255, 0, 255),
            'Dark Violet': (148, 0, 211),
            'Brown': (139, 69, 19),
            'Saddle Brown': (139, 69, 19),
            'Beige': (245, 245, 220),
            'Tan': (210, 180, 140),
            'Olive': (128, 128, 0),
            'Teal': (0, 128, 128),
            'Turquoise': (64, 224, 208),
            'Indigo': (75, 0, 130),
            'Maroon': (128, 0, 0),
            'Navy': (0, 0, 128),
            'Silver': (192, 192, 192),
            'Lavender': (230, 230, 250),
            'Coral': (255, 127, 80),
            'Salmon': (250, 128, 114),
            'Chocolate': (210, 105, 30),
        }
        # Серые (ахроматические) – особая обработка
        self.gray_names = ['Black', 'Dark Gray', 'Gray', 'Light Gray', 'Silver', 'White']
        self.gray_rgb = np.array([self.color_palette[n] for n in self.gray_names], dtype=np.float32)

        self.color_names = list(self.color_palette.keys())
        self.color_rgb = np.array(list(self.color_palette.values()), dtype=np.float32)
        rgb_uint8 = self.color_rgb.reshape(1, -1, 3).astype(np.uint8)
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)
        self.color_hsv = hsv.reshape(-1, 3).astype(np.float32)  # H[0..179], S,V[0..255]

    def find_dominant_colors(self, image):
        """K-means кластеризация -> список (имя, BGR, процент)"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pixels = image_rgb.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, self.n_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        unique, counts = np.unique(labels, return_counts=True)
        sorted_idx = np.argsort(counts)[::-1]
        centers = centers[sorted_idx].astype(int)
        counts = counts[sorted_idx]
        total = sum(counts)
        result = []
        for center, count in zip(centers, counts):
            name = self._classify_color(center)
            percent = (count / total) * 100
            bgr = center[::-1].tolist()  # RGB -> BGR
            result.append((name, bgr, percent))
        return result

    def _classify_color(self, rgb_vector):
        """Улучшенная классификация: серые отдельно, цветные – по взвешенному HSV"""
        rgb_uint8 = np.array([[rgb_vector]], dtype=np.uint8)
        hsv = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2HSV)[0][0].astype(np.float32)
        h, s, v = hsv
        SAT_THRESH = 30.0
        if s < SAT_THRESH:
            # Ахроматический – ближайший по яркости (V) среди серых
            gray_v = np.array([self.color_palette[n][0] for n in self.gray_names], dtype=np.float32)
            distances = np.abs(gray_v - v)
            idx = np.argmin(distances)
            return self.gray_names[idx]
        else:
            W_H, W_S, W_V = 2.0, 0.8, 1.0
            dH = np.abs(self.color_hsv[:, 0] - h)
            dH = np.minimum(dH, 180 - dH)
            dS = np.abs(self.color_hsv[:, 1] - s)
            dV = np.abs(self.color_hsv[:, 2] - v)
            distances = np.sqrt((W_H * dH) ** 2 + (W_S * dS) ** 2 + (W_V * dV) ** 2)
            idx = np.argmin(distances)
            return self.color_names[idx]

    def spectral_order(self, colors_list):
        """Сортировка по спектру и светлоте, группировка с отступами."""
        group_order = [
            'Reds/Pinks',
            'Oranges/Yellows',
            'Greens',
            'Blues/Purples',
            'Browns/Beiges',
            'Grays'
        ]
        groups_map = {
            'Reds/Pinks': ['Dark Red', 'Red', 'Maroon', 'Deep Pink', 'Pink', 'Light Pink', 'Coral', 'Salmon'],
            'Oranges/Yellows': ['Dark Orange', 'Orange', 'Gold', 'Yellow', 'Light Yellow', 'Chocolate'],
            'Greens': ['Dark Green', 'Green', 'Lime', 'Light Green', 'Olive', 'Teal', 'Turquoise'],
            'Blues/Purples': ['Dark Blue', 'Blue', 'Light Blue', 'Cyan', 'Purple', 'Magenta', 'Dark Violet', 'Indigo', 'Navy', 'Lavender'],
            'Browns/Beiges': ['Brown', 'Saddle Brown', 'Beige', 'Tan'],
            'Grays': ['Black', 'Dark Gray', 'Gray', 'Light Gray', 'Silver', 'White']
        }
        groups = {g: [] for g in group_order}
        for name, bgr, percent in colors_list:
            found = False
            for gname, gcolors in groups_map.items():
                if name in gcolors:
                    groups[gname].append((name, bgr, percent))
                    found = True
                    break
            if not found:
                groups.setdefault('Other', []).append((name, bgr, percent))
                if 'Other' not in group_order:
                    group_order.append('Other')

        def luminance(item):
            name, _, _ = item
            rgb = self.color_palette[name]
            return 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]

        sorted_list = []
        for g in group_order:
            if g in groups and groups[g]:
                sorted_group = sorted(groups[g], key=luminance, reverse=True)
                sorted_list.extend(sorted_group)
        return sorted_list

def create_flat_legend_image(colors_ordered, width=400, row_height=35):
    """Создаёт изображение легенды без заголовков, с отступами между группами."""
    group_boundaries = {
        'Reds/Pinks': ['Dark Red', 'Red', 'Maroon', 'Deep Pink', 'Pink', 'Light Pink', 'Coral', 'Salmon'],
        'Oranges/Yellows': ['Dark Orange', 'Orange', 'Gold', 'Yellow', 'Light Yellow', 'Chocolate'],
        'Greens': ['Dark Green', 'Green', 'Lime', 'Light Green', 'Olive', 'Teal', 'Turquoise'],
        'Blues/Purples': ['Dark Blue', 'Blue', 'Light Blue', 'Cyan', 'Purple', 'Magenta', 'Dark Violet', 'Indigo', 'Navy', 'Lavender'],
        'Browns/Beiges': ['Brown', 'Saddle Brown', 'Beige', 'Tan'],
        'Grays': ['Black', 'Dark Gray', 'Gray', 'Light Gray', 'Silver', 'White']
    }

    items_with_group = []
    for name, bgr, percent in colors_ordered:
        cur_group = None
        for g, names in group_boundaries.items():
            if name in names:
                cur_group = g
                break
        if cur_group is None:
            cur_group = 'Other'
        items_with_group.append((name, bgr, percent, cur_group))

    total_height = 20
    prev_group = None
    for _, _, _, grp in items_with_group:
        if prev_group is not None and grp != prev_group:
            total_height += 15
        total_height += row_height
        prev_group = grp
    total_height += 20

    legend = np.ones((total_height, width, 3), dtype=np.uint8) * 255
    y = 20
    prev_group = None
    for name, bgr, percent, grp in items_with_group:
        if prev_group is not None and grp != prev_group:
            cv2.line(legend, (20, y), (width-20, y), (220, 220, 220), 1)
            y += 10
        cv2.rectangle(legend, (15, y), (35, y + 20), bgr, -1)
        cv2.putText(legend, f"{name}: {percent:.1f}%", (45, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        bar_len = int(percent * 2.5)
        cv2.rectangle(legend, (45, y + 22), (45 + bar_len, y + 28), bgr, -1)
        y += row_height
        prev_group = grp
    return legend

class ScrollableLegend:
    """Окно легенды с плавной прокруткой колёсиком мыши, трекбаром и клавишами W/S"""
    def __init__(self, full_img, view_height=500):
        self.full = full_img
        self.view_h = view_height
        self.max_scroll = max(0, full_img.shape[0] - view_height)
        self.pos = 0
        cv2.namedWindow("Color Legend", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Color Legend", 400, view_height)
        cv2.createTrackbar("Scroll", "Color Legend", 0, self.max_scroll, self._on_trackbar)
        cv2.setMouseCallback("Color Legend", self._on_mouse)
        self.update()

    def _on_trackbar(self, val):
        self.pos = val
        self.update()

    def _on_mouse(self, event, x, y, flags, param):
        # Надёжная обработка колеса мыши (OpenCV 4.5+)
        if event == cv2.EVENT_MOUSEWHEEL:
            # getMouseWheelDelta возвращает положительное число при прокрутке вверх/вправо
            # и отрицательное при прокрутке вниз/влево
            delta = cv2.getMouseWheelDelta(flags)
            # Чувствительность: 1 строка = 30 пикселей
            step = 30
            self.pos = max(0, min(self.max_scroll, self.pos - delta * step // 120))
            cv2.setTrackbarPos("Scroll", "Color Legend", self.pos)
            self.update()

    def update(self):
        y1 = self.pos
        y2 = min(y1 + self.view_h, self.full.shape[0])
        visible = self.full[y1:y2, :]
        cv2.imshow("Color Legend", visible)

def get_screen_resolution():
    """Возвращает (ширина, высота) основного монитора."""
    cv2.namedWindow("tmp", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("tmp", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow("tmp", np.zeros((100, 100, 3), dtype=np.uint8))
    cv2.waitKey(1)
    rect = cv2.getWindowImageRect("tmp")
    cv2.destroyWindow("tmp")
    cv2.waitKey(1)
    return (rect[2], rect[3]) if rect else (1920, 1080)

def main():
    detector = AdvancedColorDetector(n_clusters=8)
    img_path = "kor.png"
    image = cv2.imread(img_path)
    if image is None:
        try:
            with open(img_path, "rb") as f:
                data = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        except:
            pass
    if image is None:
        print(f"Не удалось загрузить '{img_path}'")
        return

    dominant = detector.find_dominant_colors(image)
    print("Доминирующие цвета:")
    for name, bgr, pct in dominant:
        print(f"  {name}: {pct:.1f}%")

    ordered = detector.spectral_order(dominant)
    print("\nЦвета в спектральном порядке (от светлых к тёмным):")
    for name, bgr, pct in ordered:
        print(f"  {name}: {pct:.1f}%")

    full_legend = create_flat_legend_image(ordered, width=400, row_height=35)

    screen_w, screen_h = get_screen_resolution()
    win_w, win_h = int(screen_w * 0.9), int(screen_h * 0.9)
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", win_w, win_h)
    cv2.imshow("Image", image)

    scroll_legend = ScrollableLegend(full_legend, view_height=500)

    print("\nУправление легендой: колёсико мыши, трекбар, клавиши W/S, ESC для выхода.")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if cv2.getWindowProperty("Image", cv2.WND_PROP_VISIBLE) < 1 or \
           cv2.getWindowProperty("Color Legend", cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == 27:  # ESC
            break
        # Клавиши W / стрелка вверх
        if key == ord('w') or key == 82:
            scroll_legend.pos = max(0, scroll_legend.pos - 30)
            cv2.setTrackbarPos("Scroll", "Color Legend", scroll_legend.pos)
            scroll_legend.update()
        # Клавиши S / стрелка вниз
        elif key == ord('s') or key == 84:
            scroll_legend.pos = min(scroll_legend.max_scroll, scroll_legend.pos + 30)
            cv2.setTrackbarPos("Scroll", "Color Legend", scroll_legend.pos)
            scroll_legend.update()

    cv2.destroyAllWindows()
    cv2.imwrite("result_kor.png", image)
    cv2.imwrite("legend_ordered.png", full_legend)
    print("\nРезультаты сохранены: result_kor.png, legend_ordered.png")

if __name__ == "__main__":
    main()