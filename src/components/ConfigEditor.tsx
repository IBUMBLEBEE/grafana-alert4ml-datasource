import React, { ChangeEvent, useEffect } from 'react';
import { InlineField, Input, SecretInput, Card, Combobox, InlineSwitch } from '@grafana/ui';
import { DataSourcePluginOptionsEditorProps } from '@grafana/data';
import { Alert4MLDataSourceOptions, Alert4MLSecureJsonData, Alert4MLPgSecureJsonData } from '../types';

interface Props extends DataSourcePluginOptionsEditorProps<Alert4MLDataSourceOptions, Alert4MLSecureJsonData & Alert4MLPgSecureJsonData> {}

const SSL_MODE_OPTIONS = [
  { label: 'disable', value: 'disable' },
  { label: 'require', value: 'require' },
  { label: 'verify-ca', value: 'verify-ca' },
  { label: 'verify-full', value: 'verify-full' },
];

export function ConfigEditor(props: Props) {
  const { onOptionsChange, options } = props;
  const { jsonData, secureJsonFields, secureJsonData } = options;

  // Ensure pgSSLMode has a default value so it's always persisted
  useEffect(() => {
    if (jsonData.pgSSLMode === undefined || jsonData.pgSSLMode === null) {
      onOptionsChange({
        ...options,
        jsonData: { ...jsonData, pgSSLMode: 'disable' },
      });
    }
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

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

  const onTrialModeChange = (event: ChangeEvent<HTMLInputElement>) => {
    const trialMode = event.target.checked;
    onOptionsChange({
      ...options,
      jsonData: {
        ...jsonData,
        trialMode,
        // Clear PG fields when enabling trial mode
        ...(trialMode ? { pgHost: '', pgPort: undefined, pgDatabase: '', pgUser: '', pgSSLMode: 'disable' } : {}),
      },
      ...(trialMode ? {
        secureJsonFields: { ...options.secureJsonFields, pgPassword: false },
        secureJsonData: { ...options.secureJsonData, pgPassword: '' },
      } : {}),
    });
  };

  const onPgHostChange = (event: ChangeEvent<HTMLInputElement>) => {
    onOptionsChange({
      ...options,
      jsonData: { ...jsonData, pgHost: event.target.value },
    });
  };

  const onPgPortChange = (event: ChangeEvent<HTMLInputElement>) => {
    onOptionsChange({
      ...options,
      jsonData: { ...jsonData, pgPort: parseInt(event.target.value, 10) || undefined },
    });
  };

  const onPgDatabaseChange = (event: ChangeEvent<HTMLInputElement>) => {
    onOptionsChange({
      ...options,
      jsonData: { ...jsonData, pgDatabase: event.target.value },
    });
  };

  const onPgUserChange = (event: ChangeEvent<HTMLInputElement>) => {
    onOptionsChange({
      ...options,
      jsonData: { ...jsonData, pgUser: event.target.value },
    });
  };

  const onPgSSLModeChange = (option: { label?: string; value: string } | null) => {
    onOptionsChange({
      ...options,
      jsonData: { ...jsonData, pgSSLMode: option?.value },
    });
  };

  const onPgPasswordChange = (event: ChangeEvent<HTMLInputElement>) => {
    onOptionsChange({
      ...options,
      secureJsonData: {
        ...secureJsonData,
        pgPassword: event.target.value,
      },
    });
  };

  const onResetPgPassword = () => {
    onOptionsChange({
      ...options,
      secureJsonFields: {
        ...options.secureJsonFields,
        pgPassword: false,
      },
      secureJsonData: {
        ...options.secureJsonData,
        pgPassword: '',
      },
    });
  };

  return (
    <>
    <Card>
      <Card.Heading>Grafana Connection</Card.Heading>
      <Card.Description>
      <InlineField label="URL" labelWidth={14} interactive tooltip={'Grafana URL'}>
          <Input
            id="config-editor-url"
            onChange={onPathChange}
            value={jsonData.url}
            placeholder="Enter the url, e.g. http://localhost:3000"
            width={40}
          />
        </InlineField>
        <InlineField label="API Token" labelWidth={14} interactive tooltip={'Secure json field for API token'}>
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
    <Card>
      <Card.Heading>Storage</Card.Heading>
      <Card.Description>
        <InlineField label="Trial Mode" labelWidth={14} tooltip={'Enable trial mode to use SQLite in-memory storage instead of PostgreSQL'}>
          <InlineSwitch
            id="config-editor-trial-mode"
            value={jsonData.trialMode ?? false}
            onChange={onTrialModeChange}
          />
        </InlineField>
        {jsonData.trialMode && (
          <p style={{ color: '#8e8e8e', fontSize: '12px', marginTop: '4px' }}>
            Trial mode uses SQLite in-memory storage. Data will be lost when the plugin restarts.
          </p>
        )}
        {!jsonData.trialMode && (
          <>
            <h6 style={{ marginTop: '16px', marginBottom: '8px' }}>PostgreSQL Connection</h6>
            <InlineField label="Host" labelWidth={14} tooltip={'PostgreSQL host address'} required>
              <Input
                id="config-editor-pg-host"
                onChange={onPgHostChange}
                value={jsonData.pgHost ?? ''}
                placeholder="e.g. localhost"
                width={40}
                required
                invalid={!jsonData.pgHost}
              />
            </InlineField>
            <InlineField label="Port" labelWidth={14} tooltip={'PostgreSQL port'}>
              <Input
                id="config-editor-pg-port"
                type="number"
                onChange={onPgPortChange}
                value={jsonData.pgPort ?? 5432}
                placeholder="5432"
                width={40}
              />
            </InlineField>
            <InlineField label="Database" labelWidth={14} tooltip={'PostgreSQL database name'} required>
              <Input
                id="config-editor-pg-database"
                onChange={onPgDatabaseChange}
                value={jsonData.pgDatabase ?? ''}
                placeholder="e.g. alert4ml"
                width={40}
                required
                invalid={!jsonData.pgDatabase}
              />
            </InlineField>
            <InlineField label="User" labelWidth={14} tooltip={'PostgreSQL user'} required>
              <Input
                id="config-editor-pg-user"
                onChange={onPgUserChange}
                value={jsonData.pgUser ?? ''}
                placeholder="e.g. postgres"
                width={40}
                required
                invalid={!jsonData.pgUser}
              />
            </InlineField>
            <InlineField label="Password" labelWidth={14} tooltip={'PostgreSQL password'} required>
              <SecretInput
                id="config-editor-pg-password"
                isConfigured={secureJsonFields.pgPassword ?? false}
                value={secureJsonData?.pgPassword ?? ''}
                placeholder="Enter password"
                width={40}
                onReset={onResetPgPassword}
                onChange={onPgPasswordChange}
              />
            </InlineField>
            <InlineField label="SSL Mode" labelWidth={14} tooltip={'PostgreSQL SSL mode'}>
              <Combobox
                id="config-editor-pg-sslmode"
                options={SSL_MODE_OPTIONS}
                value={jsonData.pgSSLMode ?? 'disable'}
                onChange={onPgSSLModeChange}
                width={40}
              />
            </InlineField>
          </>
        )}
      </Card.Description>
    </Card>
    </>
  );
}
