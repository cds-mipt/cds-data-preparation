# cds-data-preparation

## crop_light.py
Этот скрипт используется для получения датасета из светофоров.
На вход он принимает следующие параметры:
* `folder-train`
* `folder-test`
* `attitude`
* `input-file`

`folder-train` - путь папки в которую будет записываться датасет для обучения сетки, если папка не существует, то она будет создана автоматически   
`folder-test` - путь папки в которую будет записываться датасет для валидации сетки, если папка не существует, то она будет создана автоматически   
`attitude` - отношение `train` к `train+test` (например: при `attitude=0.8` 80% информации будет в `folder-train`)   
`input-file` - файл типа `JSON` в котором находятся аннотации к городу/городам/всему_датасету   

### Пример использования
`python crop_light.py --output-folder-test /datasets/DTLD/DTLD_crop/test --output-folder-train /datasets/DTLD/DTLD_crop/train --attitude 0.8 --input-file /datasets/DTLD/JSONS/Bochum_all.json`
