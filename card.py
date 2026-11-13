MAX_TITLE_LENGTH = 70
PRESSURE_PURPOSE = "Прибор предназначен для измерения избыточного давления."
MANOMETER_FIELDS = ("диапазон измерений", "класс точности", "присоединение", "диаметр корпуса")


def title_options(product):
    name = product["name"].strip()
    manufacturer = product["manufacturer"].strip()
    product_type = product.get("productType", "").strip()
    if len(name) <= MAX_TITLE_LENGTH:
        candidates = [f"{name} {manufacturer}", name]
    else:
        candidates = [f"{product_type} {manufacturer}".strip(), manufacturer, "Карточка товара"]
    return list(dict.fromkeys(title for title in candidates if len(title) <= MAX_TITLE_LENGTH))


def filled_specs(product):
    return [(name, value) for name, value in product["specs"].items() if value.strip()]


def query_options(product):
    name = product["name"].strip()
    candidates = [name, f"{name} {product['manufacturer'].strip()}"]
    for field, label in (("productType", ""), ("article", "артикул "), ("productCode", "код товара ")):
        value = product.get(field, "")
        if value.strip():
            candidates.append(f"{name} {label}{value}".strip())
    candidates.extend(f"{name} {label} {value}".strip() for label, value in filled_specs(product))
    candidates.extend([f"{name} технические характеристики", f"{name} описание"])

    unique = {}
    for query in candidates:
        unique.setdefault(query.casefold(), query)
    return list(unique.values())


def source_description(product):
    text = (product["description"] or "").strip()
    if not text:
        return ""
    if text.endswith(("...", "…")):
        return text.rstrip(".…").rstrip() + "."
    return text if text.endswith((".", "!", "?")) else text + "."


def manager_summary(product):
    source = source_description(product)
    if not source:
        return "Данных о назначении и целевой аудитории нет."
    if source == PRESSURE_PURPOSE:
        return f"{source} Сведений о целевой аудитории нет."
    return source.split("\n\n", 1)[0]


def descriptions(product):
    intro = f"{product['name']} — изделие производителя {product['manufacturer']}."
    product_type = product.get("productType", "")
    if product_type.strip():
        intro += f" Тип товара — {product_type}."
    source = source_description(product)
    if source:
        intro += f" {source}"
    specs = " ".join(f"{name[:1].upper()}{name[1:]} — {value}." for name, value in filled_specs(product))
    options = {"facts": "\n\n".join(part for part in (intro, specs) if part)}

    values = dict(filled_specs(product))
    if source == PRESSURE_PURPOSE and values.keys() >= set(MANOMETER_FIELDS):
        pressure_range, accuracy, connection, diameter = (values[field] for field in MANOMETER_FIELDS)
        measurement = (
            f"Диапазон измерений — {pressure_range}. При подборе манометра его нужно сопоставить "
            "с давлением, которое предстоит измерять. "
            f"Класс точности — {accuracy}. Это отдельный параметр выбора. "
            "Его следует сопоставить с требованиями к точности измерения."
        )
        installation = (
            f"Присоединение — {connection}. Перед заказом обозначение присоединения нужно сверить "
            "с ответной частью в точке установки. "
            f"Диаметр корпуса — {diameter}. По этому размеру можно предварительно оценить, "
            "поместится ли корпус в отведённом месте. Для проверки всех монтажных размеров нужен чертёж."
        )
        additional_specs = " ".join(
            f"{name[:1].upper()}{name[1:]} — {value}."
            for name, value in values.items()
            if name not in MANOMETER_FIELDS
        )
        clarification = (
            "В исходном описании условия эксплуатации не указаны. Перед применением стоит уточнить "
            "допустимую рабочую среду и температуру, а также требования к монтажу в выбранной точке измерения."
        )
        options["detailed"] = "\n\n".join(
            part for part in (intro, measurement, installation, additional_specs, clarification) if part
        )

    return options
