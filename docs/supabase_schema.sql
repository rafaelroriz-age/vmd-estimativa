-- RASCUNHO (não aplicado). Esquema sugerido para migrar os dados do PWA para o Supabase.
-- O app lê tudo por web/src/data.ts (interface DataSource): basta criar um SupabaseSource
-- que devolva o mesmo formato (Seg[], GeoJSON, ClassData, Models, Walk[]).
create extension if not exists postgis;

create table sre_segmento (
  sre text not null, sentido text not null check (sentido in ('CRESCENTE','DECRESCENTE')),
  rodovia text, classe text, situacao text, perim_urb boolean, regional int,
  km_inicial numeric, km_final numeric, extensao numeric,
  media_empresas_ativas numeric, media_populacao_residente numeric, media_pib numeric,
  media_frotas_ativas numeric, distancia_cid_1 numeric,
  ano_contagem int,
  primary key (sre, sentido)
);

create table sre_geometria (sre text primary key, situacao text, geom geometry(MultiLineString, 4674));

create table estimativa (
  modelo text not null check (modelo in ('eb','car')),
  sre text not null, sentido text not null, classe text not null,   -- classe: 35 classes ou 'VMD'
  origem text not null check (origem in ('contagem','estimativa')),
  valor numeric not null, ic90_lo numeric, ic90_hi numeric,
  rodada text not null default '2026-09',
  primary key (modelo, sre, sentido, classe, rodada),
  foreign key (sre, sentido) references sre_segmento (sre, sentido)
);
create index on estimativa (rodada, classe);

-- Leitura pública apenas (RLS)
alter table sre_segmento enable row level security;
alter table sre_geometria enable row level security;
alter table estimativa enable row level security;
create policy "leitura" on sre_segmento for select using (true);
create policy "leitura" on sre_geometria for select using (true);
create policy "leitura" on estimativa for select using (true);
