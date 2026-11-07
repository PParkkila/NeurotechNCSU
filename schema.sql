-- NeuroPawn Shape Battle: OPTIONAL Supabase setup (only needed for two separate devices / headset bridge).
-- Supabase dashboard -> SQL Editor -> New query -> paste -> Run.

create extension if not exists pgcrypto;

-- clean up objects from the earlier dashboard draft, if you ran it (safe if you did not)
drop view if exists matches_public;
drop function if exists create_match(text, text, text);
drop function if exists submit_shape(text, int, text, text, jsonb, numeric);

create table if not exists matches (
  id          uuid primary key default gen_random_uuid(),
  code        text unique not null,
  p1_name     text default 'Player 1',
  p2_name     text default 'Player 2',
  p1_shape    text, p1_score numeric,
  p2_shape    text, p2_score numeric,
  status      text not null default 'waiting',   -- waiting | ready | revealed
  winner      text,                              -- p1 | p2 | tie
  created_at  timestamptz not null default now(),
  revealed_at timestamptz
);
-- RLS on with no policies: the browser can never read or write this table directly.
alter table matches enable row level security;

-- >>> RULES: EDIT HERE to change who beats whom (keep in sync with BEATS in js/config.js) <<<
create or replace function beats(a text, b text) returns boolean language sql immutable as $$
  select (a, b) in (('square','triangle'), ('triangle','circle'), ('circle','square'));
$$;

-- The site reads only this view. Shapes stay NULL until the match is revealed.
create view matches_public as
select
  code, p1_name, p2_name, status, winner, created_at, revealed_at,
  (p1_shape is not null) as p1_locked,
  (p2_shape is not null) as p2_locked,
  case when status = 'revealed' then p1_shape end as p1_shape,
  case when status = 'revealed' then p1_score end as p1_score,
  case when status = 'revealed' then p2_shape end as p2_shape,
  case when status = 'revealed' then p2_score end as p2_score
from matches;
grant select on matches_public to anon, authenticated;

create or replace function create_match(p_p1 text, p_p2 text)
returns text language plpgsql security definer set search_path = public as $$
declare v_code text;
begin
  loop
    v_code := upper(substr(md5(random()::text), 1, 5));
    exit when not exists (select 1 from matches where code = v_code);
  end loop;
  insert into matches (code, p1_name, p2_name)
  values (v_code, coalesce(nullif(left(p_p1, 24), ''), 'Player 1'), coalesce(nullif(left(p_p2, 24), ''), 'Player 2'));
  return v_code;
end $$;

-- Used by the website AND by the Python bridge. p_score = decoder confidence 0..1 (optional).
create or replace function submit_shape(p_code text, p_slot int, p_shape text, p_score numeric default null)
returns void language plpgsql security definer set search_path = public as $$
declare m matches;
begin
  select * into m from matches where code = upper(p_code);
  if not found then raise exception 'Match % not found', p_code; end if;
  if m.status = 'revealed' then raise exception 'Match already revealed'; end if;
  if p_shape not in ('square','circle','triangle') then raise exception 'Bad shape'; end if;
  if p_slot = 1 then
    update matches set p1_shape = p_shape, p1_score = p_score where code = m.code;
  elsif p_slot = 2 then
    update matches set p2_shape = p_shape, p2_score = p_score where code = m.code;
  else
    raise exception 'Slot must be 1 or 2';
  end if;
  update matches set status = 'ready' where code = m.code and p1_shape is not null and p2_shape is not null;
end $$;

create or replace function battle_match(p_code text)
returns void language plpgsql security definer set search_path = public as $$
declare m matches; w text;
begin
  select * into m from matches where code = upper(p_code);
  if not found then raise exception 'Match % not found', p_code; end if;
  if m.p1_shape is null or m.p2_shape is null then raise exception 'Both players must lock in first'; end if;
  if m.status = 'revealed' then return; end if;
  w := case when m.p1_shape = m.p2_shape then 'tie'
            when beats(m.p1_shape, m.p2_shape) then 'p1' else 'p2' end;
  update matches set status = 'revealed', winner = w, revealed_at = now() where code = m.code;
end $$;

create or replace function reset_match(p_code text)
returns void language sql security definer set search_path = public as $$
  update matches set p1_shape = null, p1_score = null, p2_shape = null, p2_score = null,
                     status = 'waiting', winner = null, revealed_at = null
  where code = upper(p_code);
$$;

grant execute on function create_match(text, text) to anon, authenticated;
grant execute on function submit_shape(text, int, text, numeric) to anon, authenticated;
grant execute on function battle_match(text) to anon, authenticated;
grant execute on function reset_match(text) to anon, authenticated;
