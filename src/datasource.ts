import { DataQueryRequest, DataSourceInstanceSettings, ScopedVars, DataQueryResponse, CoreApp } from '@grafana/data';
import { Observable } from 'rxjs';
import { getTemplateSrv, DataSourceWithBackend } from '@grafana/runtime';
import { DataQuery } from '@grafana/schema';
import {
  Alert4MLQuery,
  Alert4MLDataSourceOptions,
  Alert4MLOutputMode,
  DEFAULT_ALERT4ML_QUERY,
  DEFAULT_FUNNEL_PARAMS,
} from './types';

export class DataSource extends DataSourceWithBackend<Alert4MLQuery, Alert4MLDataSourceOptions> {
  constructor(instanceSettings: DataSourceInstanceSettings<Alert4MLDataSourceOptions>) {
    super(instanceSettings);
  }

  getDefaultQuery(app: CoreApp): Partial<Alert4MLQuery> {
    const isAlerting = app === CoreApp.UnifiedAlerting || app === CoreApp.CloudAlerting;
    return {
      ...DEFAULT_ALERT4ML_QUERY,
      outputMode: isAlerting ? Alert4MLOutputMode.AlertSignal : Alert4MLOutputMode.Visualization,
      // Return fresh nested values so editing one query cannot mutate the
      // defaults subsequently used by another query.
      hyperParams: {
        ...DEFAULT_FUNNEL_PARAMS,
        ...(isAlerting ? { evalWindowSecs: 600, alertOutputMode: 'full' as const } : {}),
      },
      targets: [],
      historyTimeRange: { ...DEFAULT_ALERT4ML_QUERY.historyTimeRange },
      uniqueKeys: { ...DEFAULT_ALERT4ML_QUERY.uniqueKeys },
    };
  }

  // Apply template variables: replace template variables in query.targets before sending the request
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
    return Array.isArray(query.targets) && query.targets.length > 0;
  }
}
