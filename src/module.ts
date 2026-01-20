import { DataSourcePlugin } from '@grafana/data';
import { DataSource } from './datasource';
import { ConfigEditor } from './components/ConfigEditor';
import { QueryEditorv2 } from './components/QueryEditorv2';
import { Alert4MLQuery, Alert4MLDataSourceOptions } from './types';

export const plugin = new DataSourcePlugin<DataSource, Alert4MLQuery, Alert4MLDataSourceOptions>(DataSource)
  .setConfigEditor(ConfigEditor)
  .setQueryEditor(QueryEditorv2);
