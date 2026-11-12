import argparse
import json
import sys
from pathlib import Path

from llm import LLMError, generate_card_text


REQUIRED_FIELDS = ("name", "manufacturer")
OPTIONAL_TEXT_FIELDS = ("description", "productType", "article", "productCode")
PRODUCT_FIELDS = {*REQUIRED_FIELDS, "specs", *OPTIONAL_TEXT_FIELDS}


class InputError(ValueError):
    def __init__(self, message, field=None):
        super().__init__(message)
        self.field = field


def reject_json_constant(value):
    raise InputError(f"Значение {value} недопустимо в JSON.")


def read_json(path):
    try:
        source = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise InputError(f"Файл «{path}» должен быть в кодировке UTF-8.") from error
    except OSError as error:
        raise InputError(f"Не удалось прочитать файл «{path}» ({error.strerror}).") from error

    try:
        return json.loads(source, parse_constant=reject_json_constant)
    except json.JSONDecodeError as error:
        raise InputError(
            f"Некорректный JSON в строке {error.lineno}, столбце {error.colno} "
            f"({error.msg})."
        ) from error
    except RecursionError as error:
        raise InputError("Слишком глубокая вложенность JSON.") from error


def prepare_product(data):
    if not isinstance(data, dict):
        raise InputError("Ожидается JSON-объект с данными одного товара.")

    for field in REQUIRED_FIELDS:
        if field not in data:
            raise InputError("Обязательное поле отсутствует.", field)
        value = data[field]
        if not isinstance(value, str):
            raise InputError("Ожидается строка.", field)
        if not value.strip():
            raise InputError("Обязательное поле не может быть пустым.", field)

    for field in data:
        if field not in PRODUCT_FIELDS:
            raise InputError("Поле не предусмотрено входным форматом.", field)

    fields_to_clarify = []
    specs = data.get("specs", {})
    if not isinstance(specs, dict):
        raise InputError("Ожидается объект с характеристиками.", "specs")
    if not specs:
        fields_to_clarify.append("specs")
    for name, value in specs.items():
        field = f"specs.{name}"
        if not name.strip():
            raise InputError("Название характеристики не может быть пустым.", "specs")
        if not isinstance(value, str):
            raise InputError("Ожидается строковое значение характеристики.", field)
        if not value.strip():
            fields_to_clarify.append(field)

    for field in OPTIONAL_TEXT_FIELDS:
        if field not in data or (field == "description" and data[field] is None):
            fields_to_clarify.append(field)
            continue
        value = data[field]
        if not isinstance(value, str):
            message = "Ожидается строка или null." if field == "description" else "Ожидается строка."
            raise InputError(message, field)
        if not value.strip():
            fields_to_clarify.append(field)

    product = {**data, "specs": specs, "description": data.get("description")}
    return {"product": product, "fieldsToClarify": fields_to_clarify}


def main():
    parser = argparse.ArgumentParser(description="Формирует черновик карточки товара из JSON.")
    parser.add_argument("path", type=Path, help="Путь к JSON-файлу в UTF-8.")
    args = parser.parse_args()

    try:
        prepared = prepare_product(read_json(args.path))
        product = prepared["product"]
        has_description = bool((product["description"] or "").strip())
        filled_spec_count = sum(1 for value in product["specs"].values() if value.strip())
        is_incomplete = not has_description and filled_spec_count <= 1
        result = {
            **generate_card_text(prepared, allow_short_description=is_incomplete),
            "specs": product["specs"],
            "fieldsToClarify": prepared["fieldsToClarify"],
        }
        if is_incomplete:
            result["isIncomplete"] = True
    except (InputError, LLMError) as error:
        details = {"message": str(error)}
        if isinstance(error, InputError) and error.field is not None:
            details["field"] = error.field
        if isinstance(error, LLMError) and error.details:
            details["details"] = error.details
        print(json.dumps({"error": details}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
