package plugin

import (
	"fmt"
	"time"

	"github.com/grafana/grafana-plugin-sdk-go/data"
)

// ConvertTimezField 应用 FieldConverter 对指定字段进行转换
func ConvertField(frame *data.Frame, fieldName string, converter data.FieldConverter) error {
	// 查找对应字段
	field, idx := frame.FieldByName(fieldName)
	if field == nil {
		return fmt.Errorf("field %s not found in the frame", fieldName)
	}

	// 应用转换器
	newValues := make([]float64, field.Len())
	for i := range field.Len() {
		value := field.At(i)
		convertedValue, err := converter.Converter(value)
		if err != nil {
			return fmt.Errorf("error converting value: %v", err)
		}
		newValues[i] = convertedValue.(float64)
	}

	// 替换原始值
	newField := data.NewField(field.Name, field.Labels, newValues)
	// 更新 frame 中的字段
	frame.Fields[idx] = newField
	return nil
}

var time2Float64FieldConverter = data.FieldConverter{ // a converter appropriate for our pretend API's Timez type.
	OutputFieldType: data.FieldTypeFloat64,
	Converter: func(v any) (any, error) {
		// 如果已经是 float64 类型，直接返回
		if val, ok := v.(float64); ok {
			return val, nil
		}
		// 如果是 time.Time 类型，转换为 Unix 时间戳
		if val, ok := v.(time.Time); ok {
			return float64(val.Unix()), nil
		}
		// 其他类型报错
		return nil, fmt.Errorf("expected time.Time or float64 input but got type %T", v)
	},
}
