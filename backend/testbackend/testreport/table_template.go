package testreport

import "text/template"

// ReportTable ...
var ReportTable = template.Must(template.New("report").Parse(ReportTemplate))

// ReportTemplate ...
const ReportTemplate = `
| Status | OpType | Test | Passed | Skipped | Failed |
|:------:|--------|------|:------:|:-------:|:------:|
{{ range . -}}
|{{template "flag" .}}|{{ .OpType }}|{{ .Title }}|{{ .Tested }}|{{ .Skipped }}|{{ .Failed }}|
{{ end }}
{{- define "flag"}}
    {{- if .Failed -}}
	{{- template "red" -}}
    {{- else -}}
	{{- if .Skipped -}}
	    {{- template "orange" -}}
	{{- else -}}
	    {{- template "green" -}}
	{{- end -}}
    {{- end -}}
{{- end -}}
{{- define "red"}}❌{{ end }}
{{- define "green"}}✅{{ end}}
{{- define "orange"}}⚠️{{end}}
`
