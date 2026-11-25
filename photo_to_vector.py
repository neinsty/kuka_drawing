import numpy as np
import shapely.geometry

# Исправляем несовместимость hatched 0.2.0 с Shapely 2.x
_orig_mls = shapely.geometry.MultiLineString.__new__
def _patched_mls(cls, lines=None):
    if isinstance(lines, np.ndarray) and lines.size == 0:
        lines = []
    return _orig_mls(cls, lines)
shapely.geometry.MultiLineString.__new__ = _patched_mls



import vpype_cli

def generate_and_preview_svg(
    input_image: str,
    output_svg: str,
    width_mm: float = 210.0,
    height_mm: float = 297.0,
    show_preview: bool = True
):
    # Формируем конвейер обработки:
    # hatched -l 4 -> создает 4 уровня штриховки (можно менять на 3..6 для детализации)
    # layout -> вписывает рисунок в лист заданного размера с отступами 10 мм
    # linemerge / linesort / linesimplify -> оптимизирует линии для робота
    # write -> сохраняет готовый SVG-файл
    pipeline = (
        f'hatched -l 4 "{input_image}" '
        f'layout --fit-to-margins 10mm {width_mm}x{height_mm}mm '
        f'linemerge --tolerance 0.5mm '
        f'linesort '
        f'linesimplify --tolerance 0.3mm '
        f'write "{output_svg}"'
    )

    # Если нужно интерактивное окно предпросмотра на экране — добавляем команду show
    if show_preview:
        pipeline += " show"

    print(f"Обрабатываю {input_image}...")
    vpype_cli.execute(pipeline)
    print(f"Готово! Файл сохранен как {output_svg}")


if __name__ == "__main__":
    generate_and_preview_svg(
        input_image="WIN_20261123_09_20_39_Pro.jpg",   # Путь к вашему исходному фото
        output_svg="result1.svg",      # Куда сохранить векторный рисунок
        width_mm=210.0,               # Ширина листа (мм)
        height_mm=297.0,              # Высота листа (мм)
        show_preview=True             # Откроет окно предпросмотра (закройте его, чтобы скрипт завершился)
    )
