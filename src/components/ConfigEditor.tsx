import React, { ChangeEvent } from 'react';
import { InlineField, Input, SecretInput, Card } from '@grafana/ui';
import { DataSourcePluginOptionsEditorProps } from '@grafana/data';
import { Alert4MLDataSourceOptions, Alert4MLSecureJsonData } from '../types';

interface Props extends DataSourcePluginOptionsEditorProps<Alert4MLDataSourceOptions, Alert4MLSecureJsonData> {}

export function ConfigEditor(props: Props) {
  const { onOptionsChange, options } = props;
  const { jsonData, secureJsonFields, secureJsonData } = options;

  const onPathChange = (event: ChangeEvent<HTMLInputElement>) => {
    onOptionsChange({
      ...options,
      jsonData: {
        ...jsonData,
        url: event.target.value,
      },
    });
  };

  // Secure field (only sent to the backend)
  const onAPITokenChange = (event: ChangeEvent<HTMLInputElement>) => {
    onOptionsChange({
      ...options,
      secureJsonData: {
        apiToken: event.target.value,
      },
    });
  };

  const onResetAPIToken = () => {
    onOptionsChange({
      ...options,
      secureJsonFields: {
        ...options.secureJsonFields,
        apiToken: false,
      },
      secureJsonData: {
        ...options.secureJsonData,
        apiToken: '',
      },
    });
  };

  return (
    <>
    <Card>
      <Card.Heading>Grafana Connection</Card.Heading>
      <Card.Description>
      <InlineField label="URL" labelWidth={14} interactive tooltip={'Json field returned to frontend'}>
          <Input
            id="config-editor-url"
            onChange={onPathChange}
            value={jsonData.url}
            placeholder="Enter the url, e.g. http://localhost:3000"
            width={40}
          />
        </InlineField>
        <InlineField label="API Token" labelWidth={14} interactive tooltip={'Secure json field (backend only)'}>
          <SecretInput
            required
            id="config-editor-api-token"
            isConfigured={secureJsonFields.apiToken}
            value={secureJsonData?.apiToken}
            placeholder="Enter your API Token"
            width={40}
            onReset={onResetAPIToken}
            onChange={onAPITokenChange}
          />
        </InlineField>
      </Card.Description>
    </Card>
    </>
  );
}
