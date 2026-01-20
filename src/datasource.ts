import { DataQueryRequest, DataSourceInstanceSettings, ScopedVars, DataQueryResponse, CoreApp } from '@grafana/data';
import { Observable } from 'rxjs';
import { getTemplateSrv, DataSourceWithBackend } from '@grafana/runtime';
import { DataQuery } from '@grafana/schema';
import { Alert4MLQuery, Alert4MLDataSourceOptions, DEFAULT_ALERT4ML_QUERY } from './types';

export class DataSource extends DataSourceWithBackend<Alert4MLQuery, Alert4MLDataSourceOptions> {
  constructor(instanceSettings: DataSourceInstanceSettings<Alert4MLDataSourceOptions>) {
    super(instanceSettings);
  }

  getDefaultQuery(_: CoreApp): Partial<Alert4MLQuery> {
    return DEFAULT_ALERT4ML_QUERY;
  }

  // 应用模板变量，在发送请求前，将 query.targets 中的模板变量替换为实际值
  applyTemplateVariables(query: Alert4MLQuery, scopedVars: ScopedVars) {
    const currentDataSourceString = JSON.stringify(query.targets);
    const currentDataSource = getTemplateSrv().replace(currentDataSourceString, scopedVars);
    const currentDataSourceObject: DataQuery[] = JSON.parse(currentDataSource);
    return {
      ...query,
      targets: currentDataSourceObject,
    };
  }
  query(request: DataQueryRequest<Alert4MLQuery>): Observable<DataQueryResponse> {
    const queries = request.targets.map((query) => {
      return {
        ...query,
        targets: query.targets,
      };
    });

    const newRequest = {
      ...request,
      targets: queries,
    };

    return super.query(newRequest);
  }

  filterQuery(query: Alert4MLQuery): boolean {
    // if no query has been provided, prevent the query from being executed
    return !!query.targets;
  }
}
